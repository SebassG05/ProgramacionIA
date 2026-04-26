import pandas as pd
import time
from pathlib import Path
import sys

from load_data import load_dataset
from filter_action_movies import filter_action_movies
from user_statistics import calculate_user_statistics
from sort_movies import sort_movies_by_rating
from convert_timestamps import convert_timestamp_to_date


def benchmark_all_operations():
    """
    Ejecuta todas las operaciones y mide sus tiempos de ejecución
    """
    print("="*70)
    print("BENCHMARK COMPLETO - PANDAS")
    print("MovieLens 20M Dataset")
    print("="*70)
    
    results = {}
    total_start = time.time()

    print("\n" + "="*70)
    print("OPERACION 1: CARGAR DATASET")
    print("="*70)
    start = time.time()
    df = load_dataset(show_info=True)
    load_time = time.time() - start
    results['1. Cargar dataset'] = load_time
    print(f"\nTiempo: {load_time:.4f} segundos")
    
    if df is None:
        print("ERROR: No se pudo cargar el dataset")
        sys.exit(1)

    print("\n" + "="*70)
    print("OPERACION 2: FILTRAR PELICULAS DE ACCION (rating >= 4)")
    print("="*70)
    start = time.time()
    filtered_df, filter_time = filter_action_movies(df, min_rating=4.0)
    results['2. Filtrar peliculas'] = filter_time
    print(f"\nTiempo: {filter_time:.4f} segundos")

    print("\n" + "="*70)
    print("OPERACION 3: CALCULAR MEDIA Y STD POR USUARIO")
    print("="*70)
    start = time.time()
    user_stats, stats_time = calculate_user_statistics(df)
    results['3. Estadisticas por usuario'] = stats_time
    print(f"\nTiempo: {stats_time:.4f} segundos")

    print("\n" + "="*70)
    print("OPERACION 4: ORDENAR PELICULAS POR RATING PROMEDIO")
    print("="*70)
    start = time.time()
    sorted_movies, sort_time = sort_movies_by_rating(df, min_ratings=50)
    results['4. Ordenar peliculas'] = sort_time
    print(f"\nTiempo: {sort_time:.4f} segundos")

    print("\n" + "="*70)
    print("OPERACION 5: CONVERTIR TIMESTAMP A FECHA")
    print("="*70)
    start = time.time()
    df_dates, convert_time = convert_timestamp_to_date(df)
    results['5. Convertir timestamps'] = convert_time
    print(f"\nTiempo: {convert_time:.4f} segundos")
    
    total_time = time.time() - total_start
    results['TOTAL'] = total_time
    
    return results, df


def print_benchmark_results(results):
    """
    Imprime un resumen de los resultados del benchmark
    """
    print("\n" + "="*70)
    print("RESUMEN DE TIEMPOS DE EJECUCION - PANDAS")
    print("="*70)

    total_operations = sum(v for k, v in results.items() if k != 'TOTAL' and k != '1. Cargar dataset')
    
    print("\nTiempos individuales:")
    print("-" * 70)
    for operation, exec_time in results.items():
        if operation != 'TOTAL':
            percentage = (exec_time / results['TOTAL']) * 100
            bar_length = int(percentage / 2)
            bar = '█' * bar_length
            print(f"{operation:35s}: {exec_time:8.4f}s  {bar} {percentage:5.1f}%")
    
    print("-" * 70)
    print(f"{'TIEMPO TOTAL':35s}: {results['TOTAL']:8.4f}s")
    print(f"{'Tiempo sin carga inicial':35s}: {total_operations:8.4f}s")

    print("\n" + "="*70)
    print("ESTADISTICAS ADICIONALES")
    print("="*70)
    
    ops_without_load = {k: v for k, v in results.items() if k != 'TOTAL' and k != '1. Cargar dataset'}
    
    print(f"\nOperacion mas rapida: {min(ops_without_load, key=ops_without_load.get)}")
    print(f"   Tiempo: {min(ops_without_load.values()):.4f}s")
    
    print(f"\nOperacion mas lenta: {max(ops_without_load, key=ops_without_load.get)}")
    print(f"   Tiempo: {max(ops_without_load.values()):.4f}s")
    
    print(f"\nTiempo promedio por operacion: {sum(ops_without_load.values())/len(ops_without_load):.4f}s")

    print("\nRanking de operaciones (de mas rapida a mas lenta):")
    print("-" * 70)
    sorted_ops = sorted(ops_without_load.items(), key=lambda x: x[1])
    for idx, (op, t) in enumerate(sorted_ops, 1):
        print(f"   {idx}. {op}: {t:.4f}s")
    
    return ops_without_load


def save_benchmark_results(results):
    """
    Guarda los resultados del benchmark en un archivo CSV
    """
    output_file = Path(__file__).parent / "data" / "benchmark_results.csv"

    df_results = pd.DataFrame([
        {'Operacion': k, 'Tiempo_segundos': v, 'Framework': 'Pandas'}
        for k, v in results.items()
    ])

    df_results['Fecha_ejecucion'] = pd.Timestamp.now()

    df_results.to_csv(output_file, index=False)
    print(f"\nResultados guardados en: {output_file.name}")
    
    return df_results


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("INICIO DEL BENCHMARK")
    print("="*70)
    print("\nEste proceso ejecutara todas las operaciones y medira sus tiempos.")
    print("Puede tardar varios minutos...")
    print("\nPresiona Ctrl+C para cancelar\n")
    
    try:

        results, df = benchmark_all_operations()

        ops_times = print_benchmark_results(results)

        df_results = save_benchmark_results(results)
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETADO EXITOSAMENTE")
        print("="*70)
        
        print("\nPROXIMOS PASOS:")
        print("  - Repetir este ejercicio con Polars (modo normal)")
        print("  - Repetir con Polars (modo LazyFrame)")
        print("  - Repetir con PySpark en Databricks")
        print("  - Comparar los resultados de los 4 frameworks")
        
        return results, df_results
        
    except KeyboardInterrupt:
        print("\n\nBenchmark cancelado por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR durante el benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results, df_results = main()
