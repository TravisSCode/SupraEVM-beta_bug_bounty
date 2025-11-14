import pandas as pd
import glob

print("🔄 Подготовка к сравнению с Monad...")

# Загружаем результаты SupraBTM
supra_8cores = pd.read_csv('processed_8_ЯДЕР.csv')
supra_4cores = pd.read_csv('processed_4_ЯДРА.csv') 
supra_16cores = pd.read_csv('processed_16_ЯДЕР.csv')

print("✅ Результаты SupraBTM загружены")
print(f"SupraBTM 8 ядер: {len(supra_8cores)} блоков, средний TPS: {supra_8cores['TPS_iBTM'].mean():.0f}")
print(f"SupraBTM 4 ядра: {len(supra_4cores)} блоков, средний TPS: {supra_4cores['TPS_iBTM'].mean():.0f}")
print(f"SupraBTM 16 ядер: {len(supra_16cores)} блоков, средний TPS: {supra_16cores['TPS_iBTM'].mean():.0f}")

# Проверяем наличие результатов Monad
monad_files = glob.glob('../monad-bench/monad/monad_*threads.log')
if monad_files:
    print(f"\n📁 Найдены файлы Monad: {monad_files}")
    print("Когда тесты Monad завершатся, запустите анализ...")
else:
    print("\n❌ Файлы Monad не найдены. Завершите тесты Monad.")

print("\n🎯 Для финального сравнения выполните:")
print("1. Дождитесь завершения тестов Monad")
print("2. Запустите: python3 analysis.py execution_time.txt monad_2pe_logs.txt")
print("3. Обновите финальный отчет")
