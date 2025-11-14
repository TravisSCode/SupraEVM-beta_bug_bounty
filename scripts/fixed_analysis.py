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
        time_columns = [col for col in df.columns if 'Time' in col or 'time' in col]
        for col in time_columns:
            df[col] = df[col].apply(convert_time)
        
        # Преобразуем другие числовые колонки
        numeric_columns = ['Block_num', 'Concurrency_level', 'Block_size']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Расчет TPS (транзакций в секунду)
        df['TPS_Seq'] = df['Block_size'] / (df[time_columns[0]] / 1000)  # делим на 1000 чтобы перевести мс в секунды
        df['TPS_iBTM'] = df['Block_size'] / (df[time_columns[1]] / 1000)
        df['Speedup'] = df[time_columns[0]] / df[time_columns[1]]
        
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
        import traceback
        traceback.print_exc()
        return None

# Анализируем все конфигурации
print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ SupraBTM")
print("=" * 50)

# Сначала посмотрим на сырые данные
print("Смотрим на сырые данные...")
try:
    with open('execution_time.txt', 'r') as f:
        lines = f.readlines()[:5]
        print("Первые 5 строк execution_time.txt:")
        for line in lines:
            print(repr(line))
except Exception as e:
    print(f"Ошибка чтения файла: {e}")

df_8cores = analyze_file('execution_time.txt', '8 ЯДЕР')
df_4cores = analyze_file('../stats_4cores/execution_time.txt', '4 ЯДРА')
df_16cores = analyze_file('../stats_16cores/execution_time.txt', '16 ЯДЕР')
