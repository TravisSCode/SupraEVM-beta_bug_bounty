import pandas as pd
import numpy as np
import re

# Функция для преобразования времени в числовой формат
def convert_time(time_str):
    if isinstance(time_str, (int, float)):
        return float(time_str)

    # Удаляем все нечисловые символы кроме точки и преобразуем в float
    time_str = str(time_str).strip()

    # Если есть "ms" - миллисекунды, оставляем как есть
    if 'ms' in time_str:
        return float(re.sub(r'[^\d.]', '', time_str))
    # Если есть "µs" - микросекунды, преобразуем в миллисекунды
    elif 'µs' in time_str:
        return float(re.sub(r'[^\d.]', '', time_str)) / 1000
    else:
        # Пробуем извлечь число
        return float(re.sub(r'[^\d.]', '', time_str))

# Функция для анализа файла результатов
def analyze_file(filename, config_name):
    try:
        # Читаем файл с разделителем табуляции
        df = pd.read_csv(filename, sep='\t', engine='python')

        # Убираем лишние пробелы в названиях колонок
        df.columns = df.columns.str.strip()

        # Проверяем структуру данных
        print(f"\nСтруктура файла {filename}:")
        print(f"Колонки: {df.columns.tolist()}")
        print(f"Первые 3 строки:")
        print(df.head(3))

        # Преобразуем время в числовой формат (миллисекунды)
        time_columns = [col for col in df.columns if 'Time' in col]
        for col in time_columns:
            df[col] = df[col].apply(convert_time)

        # Преобразуем другие числовые колонки
        numeric_columns = ['Block No', 'Threads', 'Block Size']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Расчет TPS (транзакций в секунду)
        df['TPS_Seq'] = df['Block Size'] / (df['Seq. Time'] / 1000)  # делим на 1000 чтобы перевести мс в секунды
        df['TPS_iBTM'] = df['Block Size'] / (df['iBTM Time'] / 1000)
        df['Speedup'] = df['Seq. Time'] / df['iBTM Time']

        print(f"\n=== {config_name} ===")
        print(f"Блоков обработано: {len(df)}")
        print(f"Среднее ускорение: {df['Speedup'].mean():.2f}x")
        print(f"Медианное ускорение: {df['Speedup'].median():.2f}x")
        print(f"Максимальное ускорение: {df['Speedup'].max():.2f}x")
        print(f"Средний TPS последовательный: {df['TPS_Seq'].mean():.0f}")
        print(f"Средний TPS iBTM: {df['TPS_iBTM'].mean():.0f}")
        print(f"Улучшение TPS: {((df['TPS_iBTM'].mean() / df['TPS_Seq'].mean()) - 1) * 100:.1f}%")
        
        # Сохраняем обработанные данные
        output_file = f"processed_{config_name.replace(' ', '_')}.csv"
        df.to_csv(output_file, index=False)
        print(f"Обработанные данные сохранены в: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Ошибка анализа {filename}: {e}")
        import traceback
        traceback.print_exc()
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
        ],
        'Средний TPS iBTM': [
            df_4cores['TPS_iBTM'].mean(),
            df_8cores['TPS_iBTM'].mean(),
            df_16cores['TPS_iBTM'].mean()
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Сохраняем сводную статистику
    summary_df.to_csv("summary_statistics.csv", index=False)
    print(f"\nСводная статистика сохранена в: summary_statistics.csv")
