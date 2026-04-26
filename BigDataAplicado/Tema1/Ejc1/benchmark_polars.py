import polars as pl
import time
from pathlib import Path
import sys


def load_dataset_polars(show_info=True):
    """Carga el dataset con Polars"""
    dataset_file = Path(__file__).parent / "data" / "movielens_superdataset.csv"
    
    if not dataset_file.exists():
        print(f"ERROR: No se encuentra {dataset_file}")
        return None
    
    print(f"Cargando dataset desde: {dataset_file.name}")
    print(f"Tamano del archivo: {dataset_file.stat().st_size / (1024**2):.2f} MB")
    
    df = pl.read_csv(dataset_file)
    
    print(f"Dataset cargado exitosamente!")
    
    if show_info:
        print(f"\nInformacion del dataset:")
        print(f"  Filas: {df.height:,}")
        print(f"  Columnas: {df.width}")
        print(f"  Columnas: {', '.join(df.columns)}")
        print(f"\nTipos de datos:")
        print(df.schema)
        print(f"\nPrimeras 5 filas:")
        print(df.head(5))
    
    return df


def filter_action_movies_polars(df, min_rating=4.0):
    """Filtra películas de acción con rating >= min_rating"""
    print("\n" + "="*60)
    print("FILTRADO DE PELICULAS DE ACCION")
    print("="*60)
    
    print(f"\nDataset original:")
    print(f"  Total de filas: {df.height:,}")
    
    start_time = time.time()

    print(f"\nFiltrando por genero 'Action' y rating >= {min_rating}...")
    filtered_df = df.filter(
        (pl.col("genres").str.contains("Action")) &
        (pl.col("rating") >= min_rating)
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"  Filas filtradas: {filtered_df.height:,}")
    print(f"  Reduccion: {((1 - filtered_df.height/df.height) * 100):.2f}%")
    print(f"  Rating promedio: {filtered_df['rating'].mean():.3f}")
    print(f"  Usuarios unicos: {filtered_df['userId'].n_unique():,}")
    print(f"  Peliculas unicas: {filtered_df['movieId'].n_unique():,}")
    print(f"\nTiempo de ejecucion: {elapsed_time:.4f} segundos")
    
    return filtered_df, elapsed_time


def calculate_user_statistics_polars(df):
    """Calcula estadísticas por usuario"""
    print("\n" + "="*60)
    print("CALCULAR ESTADISTICAS POR USUARIO")
    print("="*60)
    
    print(f"\nDataset original:")
    print(f"  Total de ratings: {df.height:,}")
    print(f"  Usuarios unicos: {df['userId'].n_unique():,}")
    
    start_time = time.time()

    print(f"\nCalculando media y desviacion estandar por usuario...")
    user_stats = df.group_by("userId").agg([
        pl.col("rating").mean().alias("mean_rating"),
        pl.col("rating").std().alias("std_rating"),
        pl.col("rating").count().alias("count_ratings"),
        pl.col("rating").min().alias("min_rating"),
        pl.col("rating").max().alias("max_rating")
    ])

    user_stats = user_stats.with_columns(
        pl.col("std_rating").fill_null(0)
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"  Total de usuarios: {user_stats.height:,}")
    print(f"  Media global: {user_stats['mean_rating'].mean():.3f}")
    print(f"  Tiempo de ejecucion: {elapsed_time:.4f} segundos")
    
    return user_stats, elapsed_time


def sort_movies_by_rating_polars(df, min_ratings=50):
    """Ordena películas por rating promedio"""
    print("\n" + "="*60)
    print("ORDENAR PELICULAS POR RATING PROMEDIO")
    print("="*60)
    
    print(f"\nDataset original:")
    print(f"  Total de ratings: {df.height:,}")
    print(f"  Peliculas unicas: {df['movieId'].n_unique():,}")
    
    start_time = time.time()

    print(f"\nCalculando estadisticas por pelicula...")
    movie_stats = df.group_by(["movieId", "title", "genres"]).agg([
        pl.col("rating").mean().alias("mean_rating"),
        pl.col("rating").std().alias("std_rating"),
        pl.col("rating").count().alias("count_ratings"),
        pl.col("rating").min().alias("min_rating"),
        pl.col("rating").max().alias("max_rating")
    ])

    print(f"Filtrando peliculas con minimo {min_ratings} ratings...")
    filtered_movies = movie_stats.filter(pl.col("count_ratings") >= min_ratings)

    print(f"Ordenando peliculas por rating promedio...")
    sorted_movies = filtered_movies.sort("mean_rating", descending=True)
    
    elapsed_time = time.time() - start_time
    
    print(f"  Peliculas filtradas: {sorted_movies.height:,}")
    print(f"  Rating promedio: {sorted_movies['mean_rating'].mean():.3f}")
    print(f"  Tiempo de ejecucion: {elapsed_time:.4f} segundos")
    
    return sorted_movies, elapsed_time


def convert_timestamp_to_date_polars(df):
    """Convierte timestamp a fecha"""
    print("\n" + "="*60)
    print("CONVERTIR TIMESTAMP A FECHA")
    print("="*60)
    
    print(f"\nDataset original:")
    print(f"  Total de filas: {df.height:,}")
    
    start_time = time.time()

    print(f"\nConvirtiendo timestamp a formato fecha...")
    df_converted = df.with_columns([
        pl.from_epoch("timestamp", time_unit="s").alias("rating_datetime")
    ])

    print(f"Extrayendo componentes de fecha...")
    df_converted = df_converted.with_columns([
        pl.col("rating_datetime").dt.date().alias("rating_date"),
        pl.col("rating_datetime").dt.year().alias("rating_year"),
        pl.col("rating_datetime").dt.month().alias("rating_month"),
        pl.col("rating_datetime").dt.day().alias("rating_day"),
        pl.col("rating_datetime").dt.hour().alias("rating_hour"),
        pl.col("rating_datetime").dt.weekday().alias("rating_dayofweek"),
    ])
    
    elapsed_time = time.time() - start_time
    
    print(f"  Fecha mas antigua: {df_converted['rating_datetime'].min()}")
    print(f"  Fecha mas reciente: {df_converted['rating_datetime'].max()}")
    print(f"  Tiempo de ejecucion: {elapsed_time:.4f} segundos")
    
    return df_converted, elapsed_time


def benchmark_all_operations_polars():
    """Ejecuta todas las operaciones y mide tiempos"""
    print("="*70)
    print("BENCHMARK COMPLETO - POLARS (MODO NORMAL)")
    print("MovieLens 20M Dataset")
    print("="*70)
    
    results = {}
    total_start = time.time()
    
    # Operación 1: Cargar dataset
    print("\n" + "="*70)
    print("OPERACION 1: CARGAR DATASET")
    print("="*70)
    start = time.time()
    df = load_dataset_polars(show_info=True)
    load_time = time.time() - start
    results['1. Cargar dataset'] = load_time
    print(f"\nTiempo: {load_time:.4f} segundos")
    
    if df is None:
        print("ERROR: No se pudo cargar el dataset")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("OPERACION 2: FILTRAR PELICULAS DE ACCION (rating >= 4)")
    print("="*70)
    filtered_df, filter_time = filter_action_movies_polars(df, min_rating=4.0)
    results['2. Filtrar peliculas'] = filter_time
    
    print("\n" + "="*70)
    print("OPERACION 3: CALCULAR MEDIA Y STD POR USUARIO")
    print("="*70)
    user_stats, stats_time = calculate_user_statistics_polars(df)
    results['3. Estadisticas por usuario'] = stats_time

    print("\n" + "="*70)
    print("OPERACION 4: ORDENAR PELICULAS POR RATING PROMEDIO")
    print("="*70)
    sorted_movies, sort_time = sort_movies_by_rating_polars(df, min_ratings=50)
    results['4. Ordenar peliculas'] = sort_time

    print("\n" + "="*70)
    print("OPERACION 5: CONVERTIR TIMESTAMP A FECHA")
    print("="*70)
    df_dates, convert_time = convert_timestamp_to_date_polars(df)
    results['5. Convertir timestamps'] = convert_time
    
    total_time = time.time() - total_start
    results['TOTAL'] = total_time
    
    return results, df


def print_benchmark_results_polars(results):
    """Imprime resumen de resultados"""
    print("\n" + "="*70)
    print("RESUMEN DE TIEMPOS DE EJECUCION - POLARS (MODO NORMAL)")
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

    output_file = Path(__file__).parent / "data" / "benchmark_results_polars.csv"
    df_results = pl.DataFrame({
        'Operacion': list(results.keys()),
        'Tiempo_segundos': list(results.values()),
        'Framework': ['Polars'] * len(results)
    })
    df_results.write_csv(output_file)
    print(f"\nResultados guardados en: {output_file.name}")
    
    return results


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("INICIO DEL BENCHMARK - POLARS (MODO NORMAL)")
    print("="*70)
    
    try:
        results, df = benchmark_all_operations_polars()
        print_benchmark_results_polars(results)
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETADO EXITOSAMENTE")
        print("="*70)
        
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
