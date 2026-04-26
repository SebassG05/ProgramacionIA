"""
Script de benchmark con Polars (Modo LazyFrame)
Ejercicio 1 - Tema 1 - Big Data Aplicado
"""

import polars as pl
import time
from pathlib import Path
import sys


def load_dataset_polars_lazy(show_info=True):
    """Carga el dataset con Polars LazyFrame"""
    dataset_file = Path(__file__).parent / "data" / "movielens_superdataset.csv"
    
    if not dataset_file.exists():
        print(f"ERROR: No se encuentra {dataset_file}")
        return None
    
    print(f"Cargando dataset desde: {dataset_file.name}")
    print(f"Tamano del archivo: {dataset_file.stat().st_size / (1024**2):.2f} MB")
    
    # scan_csv crea un LazyFrame (no carga los datos inmediatamente)
    df_lazy = pl.scan_csv(dataset_file)
    
    print(f"LazyFrame creado exitosamente (datos aun no cargados en memoria)")
    
    if show_info:
        # Para mostrar info, necesitamos hacer collect() de una pequeña muestra
        df_sample = df_lazy.head(5).collect()
        print(f"\nInformacion del dataset:")
        print(f"  Columnas: {', '.join(df_lazy.columns)}")
        print(f"\nTipos de datos:")
        print(df_lazy.schema)
        print(f"\nPrimeras 5 filas:")
        print(df_sample)
    
    return df_lazy


def filter_action_movies_polars_lazy(df_lazy, min_rating=4.0):
    """Filtra películas de acción con rating >= min_rating"""
    print("\n" + "="*60)
    print("FILTRADO DE PELICULAS DE ACCION (LazyFrame)")
    print("="*60)
    
    start_time = time.time()
    
    # Construir la consulta (NO se ejecuta todavía)
    print(f"\nConstruyendo consulta para filtrar por genero 'Action' y rating >= {min_rating}...")
    filtered_lazy = df_lazy.filter(
        (pl.col("genres").str.contains("Action")) &
        (pl.col("rating") >= min_rating)
    )
    
    # Ejecutar la consulta optimizada
    print(f"Ejecutando consulta optimizada...")
    filtered_df = filtered_lazy.collect()
    
    elapsed_time = time.time() - start_time
    
    print(f"  Filas filtradas: {filtered_df.height:,}")
    print(f"  Rating promedio: {filtered_df['rating'].mean():.3f}")
    print(f"  Usuarios unicos: {filtered_df['userId'].n_unique():,}")
    print(f"  Peliculas unicas: {filtered_df['movieId'].n_unique():,}")
    print(f"\nTiempo de ejecucion: {elapsed_time:.4f} segundos")
    
    return filtered_df, elapsed_time


def calculate_user_statistics_polars_lazy(df_lazy):
    """Calcula estadísticas por usuario"""
    print("\n" + "="*60)
    print("CALCULAR ESTADISTICAS POR USUARIO (LazyFrame)")
    print("="*60)
    
    start_time = time.time()
    
    # Construir la consulta de agregación
    print(f"\nConstruyendo consulta de agregacion...")
    user_stats_lazy = df_lazy.group_by("userId").agg([
        pl.col("rating").mean().alias("mean_rating"),
        pl.col("rating").std().alias("std_rating"),
        pl.col("rating").count().alias("count_ratings"),
        pl.col("rating").min().alias("min_rating"),
        pl.col("rating").max().alias("max_rating")
    ])
    
    # Rellenar NaN en std con 0
    user_stats_lazy = user_stats_lazy.with_columns(
        pl.col("std_rating").fill_null(0)
    )
    
    # Ejecutar la consulta optimizada
    print(f"Ejecutando consulta optimizada...")
    user_stats = user_stats_lazy.collect()
    
    elapsed_time = time.time() - start_time
    
    print(f"  Total de usuarios: {user_stats.height:,}")
    print(f"  Media global: {user_stats['mean_rating'].mean():.3f}")
    print(f"  Tiempo de ejecucion: {elapsed_time:.4f} segundos")
    
    return user_stats, elapsed_time


def sort_movies_by_rating_polars_lazy(df_lazy, min_ratings=50):
    """Ordena películas por rating promedio"""
    print("\n" + "="*60)
    print("ORDENAR PELICULAS POR RATING PROMEDIO (LazyFrame)")
    print("="*60)
    
    start_time = time.time()
    
    # Construir la consulta completa
    print(f"\nConstruyendo consulta de agregacion, filtrado y ordenamiento...")
    movie_stats_lazy = df_lazy.group_by(["movieId", "title", "genres"]).agg([
        pl.col("rating").mean().alias("mean_rating"),
        pl.col("rating").std().alias("std_rating"),
        pl.col("rating").count().alias("count_ratings"),
        pl.col("rating").min().alias("min_rating"),
        pl.col("rating").max().alias("max_rating")
    ]).filter(
        pl.col("count_ratings") >= min_ratings
    ).sort(
        "mean_rating", descending=True
    )
    
    # Ejecutar la consulta optimizada
    print(f"Ejecutando consulta optimizada...")
    sorted_movies = movie_stats_lazy.collect()
    
    elapsed_time = time.time() - start_time
    
    print(f"  Peliculas filtradas: {sorted_movies.height:,}")
    print(f"  Rating promedio: {sorted_movies['mean_rating'].mean():.3f}")
    print(f"  Tiempo de ejecucion: {elapsed_time:.4f} segundos")
    
    return sorted_movies, elapsed_time


def convert_timestamp_to_date_polars_lazy(df_lazy):
    """Convierte timestamp a fecha"""
    print("\n" + "="*60)
    print("CONVERTIR TIMESTAMP A FECHA (LazyFrame)")
    print("="*60)
    
    start_time = time.time()
    
    # Construir la consulta de conversión
    print(f"\nConstruyendo consulta para convertir timestamps...")
    df_converted_lazy = df_lazy.with_columns([
        pl.from_epoch("timestamp", time_unit="s").alias("rating_datetime")
    ]).with_columns([
        pl.col("rating_datetime").dt.date().alias("rating_date"),
        pl.col("rating_datetime").dt.year().alias("rating_year"),
        pl.col("rating_datetime").dt.month().alias("rating_month"),
        pl.col("rating_datetime").dt.day().alias("rating_day"),
        pl.col("rating_datetime").dt.hour().alias("rating_hour"),
        pl.col("rating_datetime").dt.weekday().alias("rating_dayofweek"),
    ])
    
    # Ejecutar la consulta optimizada
    print(f"Ejecutando consulta optimizada...")
    df_converted = df_converted_lazy.collect()
    
    elapsed_time = time.time() - start_time
    
    print(f"  Fecha mas antigua: {df_converted['rating_datetime'].min()}")
    print(f"  Fecha mas reciente: {df_converted['rating_datetime'].max()}")
    print(f"  Tiempo de ejecucion: {elapsed_time:.4f} segundos")
    
    return df_converted, elapsed_time


def benchmark_all_operations_polars_lazy():
    """Ejecuta todas las operaciones y mide tiempos"""
    print("="*70)
    print("BENCHMARK COMPLETO - POLARS (MODO LAZYFRAME)")
    print("MovieLens 20M Dataset")
    print("="*70)
    
    results = {}
    total_start = time.time()
    
    # Operación 1: Cargar dataset (crear LazyFrame)
    print("\n" + "="*70)
    print("OPERACION 1: CREAR LAZYFRAME")
    print("="*70)
    start = time.time()
    df_lazy = load_dataset_polars_lazy(show_info=True)
    load_time = time.time() - start
    results['1. Crear LazyFrame'] = load_time
    print(f"\nTiempo: {load_time:.4f} segundos")
    
    if df_lazy is None:
        print("ERROR: No se pudo crear el LazyFrame")
        sys.exit(1)
    
    # Operación 2: Filtrar películas
    print("\n" + "="*70)
    print("OPERACION 2: FILTRAR PELICULAS DE ACCION (rating >= 4)")
    print("="*70)
    filtered_df, filter_time = filter_action_movies_polars_lazy(df_lazy, min_rating=4.0)
    results['2. Filtrar peliculas'] = filter_time
    
    # Operación 3: Estadísticas por usuario
    print("\n" + "="*70)
    print("OPERACION 3: CALCULAR MEDIA Y STD POR USUARIO")
    print("="*70)
    user_stats, stats_time = calculate_user_statistics_polars_lazy(df_lazy)
    results['3. Estadisticas por usuario'] = stats_time
    
    # Operación 4: Ordenar películas
    print("\n" + "="*70)
    print("OPERACION 4: ORDENAR PELICULAS POR RATING PROMEDIO")
    print("="*70)
    sorted_movies, sort_time = sort_movies_by_rating_polars_lazy(df_lazy, min_ratings=50)
    results['4. Ordenar peliculas'] = sort_time
    
    # Operación 5: Convertir timestamp
    print("\n" + "="*70)
    print("OPERACION 5: CONVERTIR TIMESTAMP A FECHA")
    print("="*70)
    df_dates, convert_time = convert_timestamp_to_date_polars_lazy(df_lazy)
    results['5. Convertir timestamps'] = convert_time
    
    total_time = time.time() - total_start
    results['TOTAL'] = total_time
    
    return results, df_lazy


def print_benchmark_results_polars_lazy(results):
    """Imprime resumen de resultados"""
    print("\n" + "="*70)
    print("RESUMEN DE TIEMPOS DE EJECUCION - POLARS (MODO LAZYFRAME)")
    print("="*70)
    
    total_operations = sum(v for k, v in results.items() if k != 'TOTAL' and k != '1. Crear LazyFrame')
    
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
    
    # Guardar resultados
    output_file = Path(__file__).parent / "data" / "benchmark_results_polars_lazy.csv"
    df_results = pl.DataFrame({
        'Operacion': list(results.keys()),
        'Tiempo_segundos': list(results.values()),
        'Framework': ['Polars_LazyFrame'] * len(results)
    })
    df_results.write_csv(output_file)
    print(f"\nResultados guardados en: {output_file.name}")
    
    return results


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("INICIO DEL BENCHMARK - POLARS (MODO LAZYFRAME)")
    print("="*70)
    print("\nLazyFrame = Construccion de plan de ejecucion optimizado")
    print("Las operaciones no se ejecutan hasta que se llama a .collect()\n")
    
    try:
        results, df_lazy = benchmark_all_operations_polars_lazy()
        print_benchmark_results_polars_lazy(results)
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETADO EXITOSAMENTE")
        print("="*70)
        
        print("\nVentajas del LazyFrame:")
        print("  - Optimizacion automatica del plan de ejecucion")
        print("  - Reduccion de uso de memoria")
        print("  - Eliminacion de operaciones redundantes")
        print("  - Procesamiento solo de columnas necesarias")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\nBenchmark cancelado por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR durante el benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()
