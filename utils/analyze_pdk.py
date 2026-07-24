import pandas as pd

def analyze_pdk_statistics(csv_path):
    # Загружаем таблицу
    df = pd.read_csv(csv_path, sep=';')
    #print(df)
    
    # Группируем по имени PDK
    stats = df.groupby('pdk_name').agg(
        total_designs=('pdk_name', 'count'),          # Сколько всего макетов
        total_drc_errors=('drc_errors', 'sum'),       # Общее число DRC нарушений
        mean_errors=('drc_errors', 'mean'),           # В среднем нарушений на макет
        max_errors=('drc_errors', 'max')              # Максимум нарушений в одном макете
    ).reset_index()
    
    # Сортируем от самых "грязных" PDK к самым чистым
    stats = stats.sort_values(by='total_drc_errors', ascending=False)
    
    # Считаем долю ошибок для каждого PDK от общего числа
    total_errors_all = stats['total_drc_errors'].sum()
    stats['error_share_%'] = (stats['total_drc_errors'] / total_errors_all * 100).round(2)
    stats.to_csv('dataset_statistics.csv')
    
    print("\n========================= СТАТИСТИКА DRC ПО PDK =========================")
    print(stats.to_string(index=False))
    print("=========================================================================\n")

if __name__ == "__main__":
    # Замени на реальный путь к твоему CSV файлу
    analyze_pdk_statistics(r"J:\dataset_Nikita_v1\dataset\all_run_info.csv")