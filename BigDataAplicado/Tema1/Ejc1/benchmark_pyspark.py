"""
BENCHMARK COMPLETO - PYSPARK
MovieLens 20M Dataset
Comparación de tiempos de ejecución
"""

import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, count, min as spark_min, max as spark_max
from pyspark.sql.functions import from_unixtime, year, month, dayofmonth, hour, dayofweek


def get_file_size(filepath):
    """Obtiene el tamaño del archivo en MB"""
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def load_dataset_pyspark(spark, show_info=True):
    """
    Carga el dataset usando PySpark
    """
    # Obtener directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "data", "movielens_superdataset.csv")
    
    if show_info:
        print(f"Cargando dataset desde: {filepath}")
        size_mb = get_file_size(filepath)
        print(f"Tamaño del archivo: {size_mb:.2f} MB")
    
    # Cargar CSV con esquema inferido
    df = spark.read.csv(
        filepath,
        header=True,
        inferSchema=True
    )
    
    if show_info:
        print(f"\nInformación del dataset:")
        print(f"  Total de registros: {df.count():,}")
        print(f"  Columnas: {', '.join(df.columns)}")
        print(f"\nTipos de datos:")
        df.printSchema()
        print("\nPrimeras 5 filas:")
        df.show(5, truncate=True)
    
    return df


def filter_action_movies_pyspark(df):
    """
    Filtra películas de acción con rating >= 4.0
    """
    print("\n" + "="*60)
    print("FILTRADO DE PELICULAS DE ACCION (PySpark)")
    print("="*60)
    
    # Filtrar por género 'Action' y rating >= 4.0
    df_action = df.filter(
        (col("genres").contains("Action")) & 
        (col("rating") >= 4.0)
    )
    
    # Calcular estadísticas
    total_rows = df_action.count()
    avg_rating = df_action.select(avg("rating")).collect()[0][0]
    unique_users = df_action.select("userId").distinct().count()
    unique_movies = df_action.select("movieId").distinct().count()
    
    print(f"  Filas filtradas: {total_rows:,}")
    print(f"  Rating promedio: {avg_rating:.3f}")
    print(f"  Usuarios unicos: {unique_users:,}")
    print(f"  Peliculas unicas: {unique_movies:,}")
    
    return df_action


def calculate_user_statistics_pyspark(df):
    """
    Calcula la media y desviación estándar de ratings por usuario
    """
    print("\n" + "="*60)
    print("CALCULAR ESTADISTICAS POR USUARIO (PySpark)")
    print("="*60)
    
    # Agrupar por usuario y calcular estadísticas
    user_stats = df.groupBy("userId").agg(
        avg("rating").alias("mean_rating"),
        stddev("rating").alias("std_rating"),
        count("rating").alias("count_ratings"),
        spark_min("rating").alias("min_rating"),
        spark_max("rating").alias("max_rating")
    )
    
    # Calcular estadísticas globales
    total_users = user_stats.count()
    global_mean = user_stats.select(avg("mean_rating")).collect()[0][0]
    
    print(f"  Total de usuarios: {total_users:,}")
    print(f"  Media global: {global_mean:.3f}")
    
    return user_stats


def sort_movies_by_rating_pyspark(df):
    """
    Ordena películas por rating promedio (mínimo 50 ratings)
    """
    print("\n" + "="*60)
    print("ORDENAR PELICULAS POR RATING PROMEDIO (PySpark)")
    print("="*60)
    
    # Agrupar por película y calcular estadísticas
    movie_stats = df.groupBy("movieId", "title", "genres").agg(
        avg("rating").alias("mean_rating"),
        count("rating").alias("count_ratings")
    )
    
    # Filtrar películas con al menos 50 ratings
    movie_stats_filtered = movie_stats.filter(col("count_ratings") >= 50)
    
    # Ordenar por rating promedio descendente
    movies_sorted = movie_stats_filtered.orderBy(col("mean_rating").desc())
    
    # Calcular estadísticas
    total_movies = movies_sorted.count()
    avg_rating = movies_sorted.select(avg("mean_rating")).collect()[0][0]
    
    print(f"  Peliculas filtradas: {total_movies:,}")
    print(f"  Rating promedio: {avg_rating:.3f}")
    
    return movies_sorted


def convert_timestamp_to_date_pyspark(df):
    """
    Convierte timestamp a fecha legible y extrae componentes
    """
    print("\n" + "="*60)
    print("CONVERTIR TIMESTAMP A FECHA (PySpark)")
    print("="*60)
    
    # Convertir timestamp a datetime y extraer componentes
    df_dates = df.withColumn("datetime", from_unixtime(col("timestamp"))) \
                 .withColumn("year", year(from_unixtime(col("timestamp")))) \
                 .withColumn("month", month(from_unixtime(col("timestamp")))) \
                 .withColumn("day", dayofmonth(from_unixtime(col("timestamp")))) \
                 .withColumn("hour", hour(from_unixtime(col("timestamp")))) \
                 .withColumn("dayofweek", dayofweek(from_unixtime(col("timestamp"))))
    
    # Obtener fecha mínima y máxima
    min_date = df_dates.select(spark_min("datetime")).collect()[0][0]
    max_date = df_dates.select(spark_max("datetime")).collect()[0][0]
    
    print(f"  Fecha mas antigua: {min_date}")
    print(f"  Fecha mas reciente: {max_date}")
    
    return df_dates


def save_benchmark_results(results, filename="benchmark_results_pyspark.csv"):
    """Guarda los resultados del benchmark en CSV"""
    import csv
    
    # Obtener directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "data", filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Operacion', 'Tiempo (segundos)'])
        for operation, time_taken in results.items():
            writer.writerow([operation, f"{time_taken:.4f}"])
    
    print(f"\nResultados guardados en: {filename}")


def print_summary(times):
    """Imprime un resumen visual de los tiempos de ejecución"""
    print("\n" + "="*70)
    print("RESUMEN DE TIEMPOS DE EJECUCION - PYSPARK")
    print("="*70)
    
    total_time = sum(times.values())
    
    print("\nTiempos individuales:")
    print("-"*70)
    
    for i, (op, time_val) in enumerate(times.items(), 1):
        percentage = (time_val / total_time) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        print(f"{i}. {op:30s}: {time_val:7.4f}s  {bar}  {percentage:5.1f}%")
    
    print("-"*70)
    print(f"TIEMPO TOTAL                       : {total_time:7.4f}s")
    
    # Calcular tiempo sin la carga inicial
    time_without_load = sum(list(times.values())[1:])
    print(f"Tiempo sin carga inicial           : {time_without_load:7.4f}s")


def main():
    print("="*70)
    print("INICIO DEL BENCHMARK - PYSPARK")
    print("="*70)
    print("\nPySpark = Framework de procesamiento distribuido")
    print("Ejecutandose en modo local (sin cluster)")
    print()
    
    # Crear SparkSession
    print("Iniciando SparkSession...")
    spark = SparkSession.builder \
        .appName("MovieLens Benchmark") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    # Configurar nivel de log (reducir verbosidad)
    spark.sparkContext.setLogLevel("WARN")
    print("SparkSession iniciado correctamente\n")
    
    print("="*70)
    print("BENCHMARK COMPLETO - PYSPARK")
    print("MovieLens 20M Dataset")
    print("="*70)
    
    # Diccionario para almacenar tiempos
    times = {}
    
    # OPERACIÓN 1: Cargar datos
    print("\n" + "="*70)
    print("OPERACION 1: CARGAR DATOS")
    print("="*70)
    start_time = time.time()
    df = load_dataset_pyspark(spark, show_info=True)
    df.cache()  # Cachear para reutilizar
    times["Cargar datos"] = time.time() - start_time
    print(f"\nTiempo: {times['Cargar datos']:.4f} segundos")
    
    # OPERACIÓN 2: Filtrar películas de acción
    print("\n" + "="*70)
    print("OPERACION 2: FILTRAR PELICULAS DE ACCION (rating >= 4)")
    print("="*70)
    start_time = time.time()
    df_action = filter_action_movies_pyspark(df)
    times["Filtrar peliculas"] = time.time() - start_time
    print(f"\nTiempo de ejecucion: {times['Filtrar peliculas']:.4f} segundos")
    
    # OPERACIÓN 3: Calcular estadísticas por usuario
    print("\n" + "="*70)
    print("OPERACION 3: CALCULAR MEDIA Y STD POR USUARIO")
    print("="*70)
    start_time = time.time()
    user_stats = calculate_user_statistics_pyspark(df)
    times["Estadisticas por usuario"] = time.time() - start_time
    print(f"  Tiempo de ejecucion: {times['Estadisticas por usuario']:.4f} segundos")
    
    # OPERACIÓN 4: Ordenar películas por rating
    print("\n" + "="*70)
    print("OPERACION 4: ORDENAR PELICULAS POR RATING PROMEDIO")
    print("="*70)
    start_time = time.time()
    movies_sorted = sort_movies_by_rating_pyspark(df)
    times["Ordenar peliculas"] = time.time() - start_time
    print(f"  Tiempo de ejecucion: {times['Ordenar peliculas']:.4f} segundos")
    
    # OPERACIÓN 5: Convertir timestamp a fecha
    print("\n" + "="*70)
    print("OPERACION 5: CONVERTIR TIMESTAMP A FECHA")
    print("="*70)
    start_time = time.time()
    df_dates = convert_timestamp_to_date_pyspark(df)
    times["Convertir timestamps"] = time.time() - start_time
    print(f"  Tiempo de ejecucion: {times['Convertir timestamps']:.4f} segundos")
    
    # Imprimir resumen
    print_summary(times)
    
    # Guardar resultados
    save_benchmark_results(times)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETADO EXITOSAMENTE")
    print("="*70)
    
    print("\nCaracteristicas de PySpark:")
    print("  - Procesamiento distribuido (escalable a clusters)")
    print("  - Optimizacion de consultas con Catalyst")
    print("  - Ejecucion lazy (evaluacion perezosa)")
    print("  - Procesamiento en memoria")
    print("  - API SQL y DataFrame unificada")
    
    # Cerrar SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
