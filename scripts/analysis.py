#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidate SupraBTM (iBTM) vs Monad2PE benchmark analysis.

Purpose
-------
Compare two execution logs (SupraBTM and Monad2PE) sharing a common Block Number space.
Normalize to milliseconds, merge per block, filter corner cases (size==0 in both files,
and extreme outliers), compute per-block speedups, and summarize overall and by
block-size buckets.

Notes
-----
Speedup definitions (computed per block first):
  iBTM vs Seq        =  Seq / iBTM
  Monad2PE vs Seq    =  Seq / Monad2PE
  iBTM vs Monad2PE   =  Monad2PE / iBTM   ( >1 ⇒ iBTM is faster )

Bucket averages and the table “Overall” row are **weighted by Block Size** (transaction count).
Per-block arithmetic means are also reported separately for comparison.

Outlier rule (Monad-centric):
  mark a block as outlier if
    Monad2PE <= min(Seq, iBTM) / OUTLIER_FACTOR
      OR
    Monad2PE >= max(Seq, iBTM) * OUTLIER_FACTOR

Inputs (TSV)
------------
SupraBTM:  Block No, Threads, Block Size, Seq. Time, iBTM Time
Monad2PE:  Block No, Threads, Fibers, Block Size, Execution Time, [Msg]

Usage
-----
  python3 analysis_v2.py supra.txt monad.txt [block_min block_max]

Outputs
-------
- output/consolidated_th{T}_fib{F}.csv  : per-block merged data & speedups
- summary/summary_th{T}_fib{F}.txt      : pretty summary (overall + buckets)
"""

import sys
import os
import re
import pandas as pd
import numpy as np


# ===== Header Compatibility Shim (auto-normalize legacy/new headers) =====
# This wraps pd.read_csv so the script accepts both old and new column names for input logs.
# It is SAFE because it only renames known variants and leaves other columns untouched.
from functools import wraps

_legacy_to_current_map = {
    # General identifiers
    "block_num": "Block No",
    "block_number": "Block No",
    "blocknumber": "Block No",
    "block no": "Block No",
    "block_size": "Block Size",
    "blocksize": "Block Size",
    "concurrency_level": "Threads",         # old Supra logs
    "concurrency": "Threads",
    "threads": "Threads",
    "fibers": "Fibers",

    # Timing columns (Supra file)
    "seq": "Seq. Time",
    "seq_time": "Seq. Time",
    "sequential": "Seq. Time",
    "sequential_time": "Seq. Time",
    "seq.": "Seq. Time",
    "ibtm": "iBTM Time",
    "ibtm_time": "iBTM Time",
    "ibtm (ms)": "iBTM Time",               # some variants

    # Timing columns (Monad file)
    "monad2pe": "Monad2PE Time",
    "monad_2pe": "Monad2PE Time",
    "monad time": "Monad2PE Time",
    "monad2pe_time": "Monad2PE Time",
}

# Build a normalized-key view of the map so lookups survive underscores/dots/hyphens.
_legacy_to_current_map_norm = {
    (re.sub(r"[^a-z0-9]+", " ", k.strip().lower()).strip()): v
    for k, v in _legacy_to_current_map.items()
}

def _normalize_header_name(name: str) -> str:
    k = re.sub(r"[^a-z0-9]+", " ", name.strip().lower())
    k = re.sub(r"\s+", " ", k).strip()
    return k

def _apply_header_compat(df):
    cols = list(df.columns)
    mapping = {}
    for c in cols:
        key = _normalize_header_name(c)
        if key in _legacy_to_current_map_norm:
            target = _legacy_to_current_map_norm[key]
            if target not in cols:  # avoid overwriting an existing correct column
                mapping[c] = target
    if mapping:
        df = df.rename(columns=mapping)
    return df

_original_read_csv = pd.read_csv

@wraps(_original_read_csv)
def _read_csv_compat(*args, **kwargs):
    df = _original_read_csv(*args, **kwargs)
    try:
        df = _apply_header_compat(df)
    except Exception:
        # Best effort: if anything goes wrong, fall back to the original dataframe
        pass
    return df

pd.read_csv = _read_csv_compat
# ===== End Compatibility Shim =====


# ---------------------------
# Tunables
# ---------------------------
SMALL_BLOCK_MAX = 50
MEDIUM_BLOCK_MAX = 200
OUTLIER_FACTOR = 10.0  # Monad2PE marked outlier if outside [min/10, max*10] vs (Seq, iBTM)

LABEL_IBTM  = "SupraBTM"
LABEL_MONAD = "Monad2PE"

# ---------------------------
# Parsing & normalization
# ---------------------------

def parse_time_to_ms(val: object) -> float:
    """
    Convert a time literal to milliseconds.
    Accepts: "...ms", "...us"/"µs", "...ns", "...s".
    Bare numbers are treated as milliseconds (common in these logs).
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(" ", "")
    try:
        if s.endswith("ms"):
            return float(s[:-2])
        if s.endswith("µs") or s.endswith("us"):
            return float(s[:-2]) / 1000.0
        if s.endswith("ns"):
            return float(s[:-2]) / 1_000_000.0
        if s.endswith("s"):
            return float(s[:-1]) * 1000.0
        return float(s)  # assume ms
    except Exception:
        return np.nan


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and trailing periods from column names."""
    df = df.copy()
    df.columns = [c.strip().rstrip('.') for c in df.columns]
    return df


def map_supra_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map SupraBTM TSV to canonical columns and parse times."""
    df = normalize_columns(df)
    expected = ["Block No", "Threads", "Block Size", "Seq. Time", "iBTM Time"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        print("\n[Header Error] Missing columns in SupraBTM file:", ", ".join(missing))
        print("Expected: Block No\tThreads\tBlock Size\tSeq. Time\tiBTM Time")
        sys.exit(1)

    df = df.rename(columns={
        "Block No": "Block Number",
        "Threads": "Threads",
        "Block Size": "Block Size",
        "Seq. Time": "Seq (raw)",
        "iBTM Time": "SupraBTM (raw)"
    })

    df["Seq (ms)"] = df["Seq (raw)"].apply(parse_time_to_ms)
    df["SupraBTM (ms)"] = df["SupraBTM (raw)"].apply(parse_time_to_ms)
    df["Block Number"] = pd.to_numeric(df["Block Number"], errors="coerce")
    df["Block Size"] = pd.to_numeric(df["Block Size"], errors="coerce")
    df["Threads"] = pd.to_numeric(df["Threads"], errors="coerce").astype("Int64")
    df["Fibers"] = pd.NA  # not present in this file
    return df[["Block Number", "Block Size", "Threads", "Fibers", "Seq (ms)", "SupraBTM (ms)"]]


def map_monad_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Monad2PE TSV to canonical columns and parse times."""
    df = normalize_columns(df)
    expected = ["Block No", "Threads", "Fibers", "Block Size", "Execution Time"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print("\n[Header Error] Missing columns in Monad2PE file:", ", ".join(missing))
        print("Expected: Block No\tThreads\tFibers\tBlock Size\tExecution Time\tMsg(optional)")
        sys.exit(1)

    colmap = {
        "Block No": "Block Number",
        "Threads": "Threads_monad",
        "Fibers": "Fibers_monad",
        "Block Size": "Block Size Monad",
        "Execution Time": "Monad2PE (raw)"
    }
    if "Msg" in df.columns:
        colmap["Msg"] = "Monad Error"
    df = df.rename(columns=colmap)

    df["Monad2PE (ms)"] = df["Monad2PE (raw)"].apply(lambda x: parse_time_to_ms(str(x).replace("us", "µs")))
    df["Block Number"] = pd.to_numeric(df["Block Number"], errors="coerce")
    df["Block Size Monad"] = pd.to_numeric(df["Block Size Monad"], errors="coerce")
    df["Threads_monad"] = pd.to_numeric(df["Threads_monad"], errors="coerce").astype("Int64")
    df["Fibers_monad"] = pd.to_numeric(df["Fibers_monad"], errors="coerce").astype("Int64")
    if "Monad Error" not in df.columns:
        df["Monad Error"] = ""
    return df[["Block Number", "Threads_monad", "Fibers_monad", "Block Size Monad", "Monad2PE (ms)", "Monad Error"]]

# ---------------------------
# Filtering & bucketing
# ---------------------------

def detect_outliers(df: pd.DataFrame, factor: float = OUTLIER_FACTOR) -> list[int]:
    """
    Flag a block as outlier if Monad2PE is implausible vs both Seq and iBTM:
      Monad2PE <= min(Seq, iBTM) / factor
      OR
      Monad2PE >= max(Seq, iBTM) * factor
    """
    outliers = []
    for _, r in df.iterrows():
        s, b, m = r["Seq (ms)"], r["SupraBTM (ms)"], r["Monad2PE (ms)"]
        if any(pd.isna([s, b, m])) or min(s, b, m) <= 0:
            continue
        lo, hi = min(s, b), max(s, b)
        if m <= lo / factor or m >= hi * factor:
            outliers.append(int(r["Block Number"]))
    return sorted(set(outliers))


def bucket_label(size: float) -> str:
    """Return the bucket label for a given Block Size."""
    if size < SMALL_BLOCK_MAX:
        return f"<{SMALL_BLOCK_MAX}"
    elif size <= MEDIUM_BLOCK_MAX:
        return f"{SMALL_BLOCK_MAX}-{MEDIUM_BLOCK_MAX}"
    else:
        return f">{MEDIUM_BLOCK_MAX}"

# ---------------------------
# Main analysis pipeline
# ---------------------------

def process_files(file1: str, file2: str, block_min=None, block_max=None) -> None:
    # Read TSVs (permit '-' comment lines)
    try:
        supra_raw = pd.read_csv(file1, sep=None, engine="python", comment="-", encoding="utf-8-sig")
        monad_raw = pd.read_csv(file2, sep=None, engine="python", comment="-", encoding="utf-8-sig")
    except Exception as e:
        print("[File Read Error] Could not read file(s):", e)
        sys.exit(1)

    supra = map_supra_columns(supra_raw)
    monad = map_monad_columns(monad_raw)

    # Optional range filter (pre-merge)
    if block_min is not None:
        supra = supra[supra["Block Number"] >= block_min]
        monad = monad[monad["Block Number"] >= block_min]
    if block_max is not None:
        supra = supra[supra["Block Number"] <= block_max]
        monad = monad[monad["Block Number"] <= block_max]

    # Remove blocks with size==0 in BOTH files
    zeros_supra = set(supra.loc[supra["Block Size"] == 0, "Block Number"].dropna().astype(int))
    zeros_monad = set(monad.loc[monad["Block Size Monad"] == 0, "Block Number"].dropna().astype(int))
    zero_blocks = sorted(list(zeros_supra & zeros_monad))

    # Merge on Block Number
    df = pd.merge(supra, monad, on="Block Number", how="inner")
    if df.empty:
        print("[Error] No overlapping blocks found between files.")
        sys.exit(1)

    if zero_blocks:
        df = df[~df["Block Number"].isin(zero_blocks)]

    # Exclude outliers (see rule above)
    outlier_blocks = detect_outliers(df)
    if outlier_blocks:
        print(f"\n[Warning] {len(outlier_blocks)} outlier blocks excluded (Monad2PE deviates ≥{OUTLIER_FACTOR}×).")
        print("Outlier Blocks:", ", ".join(map(str, outlier_blocks)))
        df = df[~df["Block Number"].isin(outlier_blocks)]

    # Fill Threads/Fibers from either side (prefer Supra fields)
    df["Threads"] = df["Threads"].where(df["Threads"].notna(), df["Threads_monad"]).astype("Int64")
    df["Fibers"]  = df["Fibers"].where(df["Fibers"].notna(), df["Fibers_monad"]).astype("Int64")

    # Per-block speedups (authoritative)
    df["SupraBTM vs Seq (×)"]      = df["Seq (ms)"] / df["SupraBTM (ms)"]
    df["Monad2PE vs Seq (×)"]      = df["Seq (ms)"] / df["Monad2PE (ms)"]
    df["SupraBTM vs Monad2PE (×)"] = df["Monad2PE (ms)"] / df["SupraBTM (ms)"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Winner
    def winner(r):
        a, b = r["SupraBTM (ms)"], r["Monad2PE (ms)"]
        if pd.isna(a) or pd.isna(b):
            return "N/A"
        if a < b: return LABEL_IBTM
        if b < a: return LABEL_MONAD
        return "Tie"

    df["Winner"] = df.apply(winner, axis=1)
    df["Bucket"] = df["Block Size"].apply(bucket_label)

    # Output files
    os.makedirs("output", exist_ok=True)
    os.makedirs("summary", exist_ok=True)
    threads_val = int(df["Threads"].dropna().iloc[0])
    fibers_val  = int(df["Fibers"].dropna().iloc[0])
    out_csv = os.path.join("output",  f"consolidated_th{threads_val}_fib{fibers_val}.csv")
    out_txt = os.path.join("summary", f"summary_th{threads_val}_fib{fibers_val}.txt")

    df_out = df[[
        "Block Number", "Block Size", "Threads", "Fibers",
        "Seq (ms)", "SupraBTM (ms)", "Monad2PE (ms)",
        "SupraBTM vs Seq (×)", "Monad2PE vs Seq (×)", "SupraBTM vs Monad2PE (×)",
        "Winner", "Bucket"
    ]].round(6)

    # Carry Monad Error if present (useful for debugging)
    if "Monad Error" in df.columns:
        df_out["Monad Error"] = df["Monad Error"]

    df_out.to_csv(out_csv, index=False)
    print(f"\nConsolidated file saved as {out_csv}")

    # ---------------------------
    # Aggregates
    # ---------------------------
    perblock_i_vs_seq = df_out["SupraBTM vs Seq (×)"].mean()
    perblock_m_vs_seq = df_out["Monad2PE vs Seq (×)"].mean()
    perblock_i_vs_m   = df_out["SupraBTM vs Monad2PE (×)"].mean()

    # Tx-weighted means (these match the bucket table “Overall” row)
    total_txs = float(df_out["Block Size"].sum())
    if total_txs > 0:
        gw_i_vs_seq = float((df_out["Block Size"] * df_out["SupraBTM vs Seq (×)"]).sum() / total_txs)
        gw_m_vs_seq = float((df_out["Block Size"] * df_out["Monad2PE vs Seq (×)"]).sum() / total_txs)
        gw_i_vs_m   = float((df_out["Block Size"] * df_out["SupraBTM vs Monad2PE (×)"]).sum() / total_txs)
    else:
        gw_i_vs_seq = gw_m_vs_seq = gw_i_vs_m = 0.0

    # Best / Worst (guard against all-NaN)
    def best_worst(series_name: str):
        s = df_out[series_name]
        s = s.dropna()
        if s.empty:
            return None, None
        i_best = s.idxmax()
        i_worst = s.idxmin()
        return df_out.loc[i_best], df_out.loc[i_worst]

    sup_best, sup_worst = best_worst("SupraBTM vs Seq (×)")
    mon_best, mon_worst = best_worst("Monad2PE vs Seq (×)")
    supmon_best, supmon_worst = best_worst("SupraBTM vs Monad2PE (×)")

    # ---------------------------
    # Bucket table (Tx-weighted averages)
    # ---------------------------

    def fmt_int(n):  # thousands separator
        return f"{int(n):,}"

    ordered_buckets = [f"<{SMALL_BLOCK_MAX}", f"{SMALL_BLOCK_MAX}-{MEDIUM_BLOCK_MAX}", f">{MEDIUM_BLOCK_MAX}"]
    bucket_rows = []
    for b in ordered_buckets:
        sub = df_out[df_out["Bucket"] == b]
        n_blks = len(sub)
        txs_sum = int(sub["Block Size"].sum())
        if n_blks == 0 or txs_sum <= 0:
            bucket_rows.append({
                "Bucket": b, "Blks": 0, "Txs": 0,
                "iWin": 0, "mWin": 0, "i%": 0.0, "m%": 0.0,
                "iSeq": 0.0, "mSeq": 0.0, "i_m": 0.0
            })
            continue

        i_wins = int((sub["Winner"] == LABEL_IBTM).sum())
        m_wins = int((sub["Winner"] == LABEL_MONAD).sum())
        i_rate = i_wins / n_blks * 100.0
        m_rate = m_wins / n_blks * 100.0

        # Weighted by Block Size (headers remain neutral: “Avg …”)
        i_vs_seq = float((sub["Block Size"] * sub["SupraBTM vs Seq (×)"]).sum() / txs_sum)
        m_vs_seq = float((sub["Block Size"] * sub["Monad2PE vs Seq (×)"]).sum() / txs_sum)
        i_vs_m   = float((sub["Block Size"] * sub["SupraBTM vs Monad2PE (×)"]).sum() / txs_sum)

        bucket_rows.append({
            "Bucket": b, "Blks": n_blks, "Txs": txs_sum,
            "iWin": i_wins, "mWin": m_wins, "i%": i_rate, "m%": m_rate,
            "iSeq": i_vs_seq, "mSeq": m_vs_seq, "i_m": i_vs_m
        })

    # Overall from buckets (Tx-weighted)
    total_blocks = sum(r["Blks"] for r in bucket_rows)
    total_txs_buckets = sum(r["Txs"] for r in bucket_rows)
    if total_txs_buckets > 0:
        w_i_vs_seq = sum(r["Txs"] * r["iSeq"] for r in bucket_rows) / total_txs_buckets
        w_m_vs_seq = sum(r["Txs"] * r["mSeq"] for r in bucket_rows) / total_txs_buckets
        w_i_vs_m   = sum(r["Txs"] * r["i_m"]  for r in bucket_rows) / total_txs_buckets
    else:
        w_i_vs_seq = w_m_vs_seq = w_i_vs_m = 0.0

    # ---------------------------
    # Print preview & summary (formatted)
    # ---------------------------
    print("\n----- Blocks (Preview) -----")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 140):
        print(df_out.drop(columns=["Bucket"]).head(20).to_string(index=False))
        #print(df_out.drop(columns=["Bucket"]).to_string(index=False))

    # helpers
    def dash(n=96): return "-" * n
    def fmt_f(x, p=3): return f"{x:.{p}f}"

    supra_wins = int((df_out["Winner"] == LABEL_IBTM).sum())
    monad_wins = int((df_out["Winner"] == LABEL_MONAD).sum())
    ties       = int((df_out["Winner"] == "Tie").sum())


    # Compact formatter for discarded lists
    def format_block_list(values, max_items=30):
        if not values:
            return "[]"
        if len(values) <= max_items:
            return "[" + ", ".join(map(str, values)) + "]"
        shown = ", ".join(map(str, values[:max_items]))
        return f"[{shown}, ... +{len(values) - max_items} more]"

    lines = []
    lines.append(dash())
    lines.append("----- SUMMARY -----")
    lines.append(dash())
    # Discarded information (counts + lists)
    lines.append("Discarded")
    lines.append(f"{'Blocks with Size==0':<26}: {len(zero_blocks)} {format_block_list(zero_blocks)}")
    lines.append(f"{'Performance Outliers':<26}: {len(outlier_blocks)} {format_block_list(outlier_blocks)}")
    lines.append(dash())

    lines.append(f"{'Total Blocks Analyzed':<26}: {len(df_out)}")
    lines.append(f"{'Total Transactions':<26}: {fmt_int(total_txs)}")
    lines.append(f"{'Average Block Size':<26}: {fmt_f(df_out['Block Size'].mean())}")
    lines.append(f"{'Average Seq (ms)':<26}: {fmt_f(df_out['Seq (ms)'].mean())}")
    lines.append(f"{'Average '+LABEL_IBTM+' (ms)':<26}: {fmt_f(df_out['SupraBTM (ms)'].mean())}")
    lines.append(f"{'Average '+LABEL_MONAD+' (ms)':<26}: {fmt_f(df_out['Monad2PE (ms)'].mean())}")
    lines.append(dash())

    # Compute avg speedups as ratio of overall averages (not avg of per-block ratios)
    _avg_block_size = df_out['Block Size'].mean()
    _avg_seq = df_out['Seq (ms)'].mean()
    _avg_ib  = df_out['SupraBTM (ms)'].mean()
    _avg_mn  = df_out['Monad2PE (ms)'].mean()

    tps_seq   = (_avg_block_size * 1000) / _avg_seq
    tps_ibtm  = (_avg_block_size * 1000) / _avg_ib
    tps_monad = (_avg_block_size * 1000) / _avg_mn

    _r_ib_seq = (_avg_seq / _avg_ib) if (_avg_ib and _avg_ib > 0) else float('nan')
    _r_mn_seq = (_avg_seq / _avg_mn) if (_avg_mn and _avg_mn > 0) else float('nan')
    _r_ib_mn  = (_avg_mn / _avg_ib) if (_avg_ib and _avg_mn and _avg_ib > 0 and _avg_mn > 0) else float('nan')

    # Per-block means
    lines.append("Average Speedup (execution time)")
    lines.append(f"{'Avg '+LABEL_IBTM+' vs Seq (×)':<30}: {fmt_f(_r_ib_seq)} (Seq Exe Time/"+LABEL_IBTM+" Exe Time)")
    lines.append(f"{'Avg '+LABEL_MONAD+' vs Seq (×)':<30}: {fmt_f(_r_mn_seq)} (Seq Exe Time/"+LABEL_MONAD+" Exe Time)")
    lines.append(f"{'Avg '+LABEL_IBTM+' vs '+LABEL_MONAD+' (×)':<30}: {fmt_f(_r_ib_mn)} ("+LABEL_MONAD+" Exe Time/"+LABEL_IBTM+" Exe Time)")
    lines.append(dash())

    # Tx-weighted means
    lines.append("Average Speedup (weighted by transactions)")
    lines.append(f"{'Avg '+LABEL_IBTM+' vs Seq (×)':<30}: {fmt_f(gw_i_vs_seq)}")
    lines.append(f"{'Avg '+LABEL_MONAD+' vs Seq (×)':<30}: {fmt_f(gw_m_vs_seq)}")
    lines.append(f"{'Avg '+LABEL_IBTM+' vs '+LABEL_MONAD+' (×)':<30}: {fmt_f(gw_i_vs_m)}")
    lines.append(dash())

    # Average throughput
    lines.append("Average Throughput (TPS)")
    lines.append(f"{'Seq':<30}: {tps_seq:,.0f} TPS")
    lines.append(f"{'SupraBTM':<30}: {tps_ibtm:,.0f} TPS")
    lines.append(f"{'Monad-2PE':<30}: {tps_monad:,.0f} TPS")
    lines.append(dash())

    # Bucket table (Tx-weighted)
    lines.append("Bucketed Results (averages weighted by Block Size)")
    hdr = (
        f"{'Bucket':<10}"
        f"{'Blocks':>8}  "
        f"{'Total Tx':>10}  "
        f"{LABEL_IBTM+' Wins':>14}  "
        f"{LABEL_MONAD+' Wins':>14}  "
        f"{LABEL_IBTM+' Win %':>12}  "
        f"{LABEL_MONAD+' Win %':>14}  "
        f"{'Avg '+LABEL_IBTM+'/Seq (×)':>22}  "
        f"{'Avg '+LABEL_MONAD+'/Seq (×)':>22}  "
        f"{'Avg '+LABEL_IBTM+'/'+LABEL_MONAD+' (×)':>26}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in bucket_rows:
        lines.append(
            f"{r['Bucket']:<10}"
            f"{r['Blks']:>8}  "
            f"{fmt_int(r['Txs']):>10}  "
            f"{r['iWin']:>14}  "
            f"{r['mWin']:>14}  "
            f"{r['i%']:>12.1f}  "
            f"{r['m%']:>14.1f}  "
            f"{r['iSeq']:>22.3f}  "
            f"{r['mSeq']:>22.3f}  "
            f"{r['i_m']:>26.3f}"
        )
    lines.append("-" * len(hdr))
    lines.append(
        f"{'Overall':<10}"
        f"{total_blocks:>8}  "
        f"{fmt_int(total_txs_buckets):>10}  "
        f"{'':>14}  "
        f"{'':>14}  "
        f"{'':>12}  "
        f"{'':>14}  "
        f"{w_i_vs_seq:>22.3f}  "
        f"{w_m_vs_seq:>22.3f}  "
        f"{w_i_vs_m:>26.3f}"
    )
    lines.append(dash())

    # Wins
    lines.append(f"{LABEL_IBTM+' Wins':<26}: {supra_wins}")
    lines.append(f"{LABEL_MONAD+' Wins':<26}: {monad_wins}")
    lines.append(f"{'Ties':<26}: {ties}")
    lines.append(dash())

    # Best / Worst section
    if all(x is not None for x in (sup_best, sup_worst, mon_best, mon_worst, supmon_best, supmon_worst)):
        lines.append("Best / Worst (per-block)")
        lines.append(
            f"{'Best '+LABEL_IBTM+' vs Seq (×)':<32}: {fmt_f(sup_best['SupraBTM vs Seq (×)'])} "
            f"(Block {int(sup_best['Block Number'])})"
        )
        lines.append(
            f"{'Worst '+LABEL_IBTM+' vs Seq (×)':<32}: {fmt_f(sup_worst['SupraBTM vs Seq (×)'])} "
            f"(Block {int(sup_worst['Block Number'])})"
        )
        lines.append(
            f"{'Best '+LABEL_MONAD+' vs Seq (×)':<32}: {fmt_f(mon_best['Monad2PE vs Seq (×)'])} "
            f"(Block {int(mon_best['Block Number'])})"
        )
        lines.append(
            f"{'Worst '+LABEL_MONAD+' vs Seq (×)':<32}: {fmt_f(mon_worst['Monad2PE vs Seq (×)'])} "
            f"(Block {int(mon_worst['Block Number'])})"
        )
        lines.append(
            f"{'Best '+LABEL_IBTM+' vs '+LABEL_MONAD+' (×)':<32}: {fmt_f(supmon_best['SupraBTM vs Monad2PE (×)'])} "
            f"(Block {int(supmon_best['Block Number'])})"
        )
        lines.append(
            f"{'Worst '+LABEL_IBTM+' vs '+LABEL_MONAD+' (×)':<32}: {fmt_f(supmon_worst['SupraBTM vs Monad2PE (×)'])} "
            f"(Block {int(supmon_worst['Block Number'])})"
        )
        lines.append(dash())

    pretty = "\n".join(lines)
    print(pretty)
    with open(out_txt, "w") as f:
        f.write(pretty)
    print(f"Summary file saved as {out_txt}\n\n")



# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        print("Usage: python analysis.py <SupraBTM_file> <Monad2PE_file> [block_min block_max]")
        sys.exit(1)
    file1, file2 = sys.argv[1], sys.argv[2]
    bmin = int(sys.argv[3]) if len(sys.argv) == 5 else None
    bmax = int(sys.argv[4]) if len(sys.argv) == 5 else None
    process_files(file1, file2, bmin, bmax)
