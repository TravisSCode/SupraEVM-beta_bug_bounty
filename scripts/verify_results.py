#!/usr/bin/env python3
"""
SupraBTM Bounty Results Verification Script
"""

import pandas as pd
import os
import sys

def main():
    print("🔍 SupraBTM Bounty Results Verification")
    print("=" * 50)
    
    # Check critical files
    files_to_check = [
        'SUPRA_BOUNTY_SUBMISSION.md',
        'data/processed/summary_statistics.csv',
        'data/processed/processed_8_ЯДЕР.csv',
        'data/processed/processed_4_ЯДРА.csv',
        'data/processed/processed_16_ЯДЕР.csv',
        'data/raw/execution_time.txt'
    ]
    
    print("📁 Checking required files...")
    for file in files_to_check:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            return False
    
    # Verify data integrity
    print("\n📊 Verifying data integrity...")
    try:
        summary = pd.read_csv('data/processed/summary_statistics.csv')
        supra_8 = pd.read_csv('data/processed/processed_8_ЯДЕР.csv')
        supra_4 = pd.read_csv('data/processed/processed_4_ЯДРА.csv')
        supra_16 = pd.read_csv('data/processed/processed_16_ЯДЕР.csv')
        
        print(f"   ✅ Summary data: {len(summary)} configurations")
        print(f"   ✅ 8-core data: {len(supra_8)} blocks")
        print(f"   ✅ 4-core data: {len(supra_4)} blocks") 
        print(f"   ✅ 16-core data: {len(supra_16)} blocks")
        
        # Verify key metrics
        max_speedup = summary['Среднее ускорение'].max()
        max_tps = summary['Макс TPS iBTM'].max()
        
        print(f"\n🎯 Key Performance Metrics:")
        print(f"   Maximum Speedup: {max_speedup:.2f}x")
        print(f"   Maximum TPS: {max_tps:.0f}")
        
        if max_speedup >= 3.0:
            print("   ✅ Speedup exceeds bounty requirements")
        else:
            print("   ⚠️  Speedup below expectations")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Data verification failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
