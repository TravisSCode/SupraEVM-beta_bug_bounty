import pandas as pd
import numpy as np

# Функция для анализа файла результатов
def analyze_file(filename, config_name):
    try:
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['Block_num', 'Concurrency_level', 'Block_size', 'Seq_Time', 'iBTM_Time']
        
        # Конвертируем время в секунды (если в миллисекундах)
        df['Seq_Time_sec'] = df['Seq_Time'] / 1000
        df['iBTM_Time_sec'] = df['iBTM_Time'] / 1000
        
        # Расчет TPS
        df['TPS_Seq'] = df['Block_size'] / df['Seq_Time_sec']
        df['TPS_iBTM'] = df['Block_size'] / df['iBTM_Time_sec']
        df['Speedup'] = df['Seq_Time'] / df['iBTM_Time']
        
        print(f"\n=== {config_name} ===")
        print(f"Блоков обработано: {len(df)}")
        print(f"Среднее ускорение: {df['Speedup'].mean():.2f}x")
        print(f"Медианное ускорение: {df['Speedup'].median():.2f}x")
        print(f"Максимальное ускорение: {df['Speedup'].max():.2f}x")
        print(f"Средний TPS последовательный: {df['TPS_Seq'].mean():.0f}")
        print(f"Средний TPS iBTM: {df['TPS_iBTM'].mean():.0f}")
        print(f"Улучшение TPS: {((df['TPS_iBTM'].mean() / df['TPS_Seq'].mean()) - 1) * 100:.1f}%")
        
        return df
    except Exception as e:
        print(f"Ошибка анализа {filename}: {e}")
        return None

# Анализируем все конфигурации
print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ SupraBTM")
print("=" * 50)

df_8cores = analyze_file('execution_time.txt', '8 ЯДЕР')
df_4cores = analyze_file('../stats_4cores/execution_time.txt', '4 ЯДРА')
df_16cores = analyze_file('../stats_16cores/execution_time.txt', '16 ЯДЕР')

# Сводная статистика
if df_8cores is not None and df_4cores is not None and df_16cores is not None:
    print("\n" + "="*50)
    print("📈 СВОДНАЯ СТАТИСТИКА")
    print("="*50)
    
    summary_data = {
        'Конфигурация': ['4 ядра', '8 ядер', '16 ядер'],
        'Блоков': [len(df_4cores), len(df_8cores), len(df_16cores)],
        'Среднее ускорение': [
            df_4cores['Speedup'].mean(),
            df_8cores['Speedup'].mean(), 
            df_16cores['Speedup'].mean()
        ],
        'Макс TPS iBTM': [
            df_4cores['TPS_iBTM'].max(),
            df_8cores['TPS_iBTM'].max(),
            df_16cores['TPS_iBTM'].max()
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
