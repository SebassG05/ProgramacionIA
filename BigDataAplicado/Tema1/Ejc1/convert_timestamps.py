import pandas as pd
import time
from pathlib import Path
from load_data import load_dataset


def convert_timestamp_to_date(df):
    """
    Convierte la columna timestamp a formato fecha/hora legible
    
    Args:
        df (pd.DataFrame): Dataset completo
        
    Returns:
        pd.DataFrame: Dataset con columnas de fecha adicionales
        float: Tiempo de ejecución
    """
    print("\n" + "="*60)
    print("CONVERSION DE TIMESTAMP A FECHA")
    print("="*60)
    
    print(f"\nDataset original:")
    print(f"   Total de filas: {len(df):,}")
    print(f"   Columnas: {', '.join(df.columns.tolist())}")

    print(f"\nInformacion del timestamp:")
    print(f"   Tipo de dato: {df['timestamp'].dtype}")
    print(f"   Valor minimo: {df['timestamp'].min()}")
    print(f"   Valor maximo: {df['timestamp'].max()}")

    print(f"\nEjemplos de timestamp original:")
    print(df[['userId', 'movieId', 'rating', 'timestamp', 'title']].head(3).to_string(index=False))

    start_time = time.time()

    df_converted = df.copy()

    print(f"\nConvirtiendo timestamp a formato fecha...")
    df_converted['rating_datetime'] = pd.to_datetime(df_converted['timestamp'], unit='s')

    print(f"Extrayendo componentes de fecha...")
    df_converted['rating_date'] = df_converted['rating_datetime'].dt.date
    df_converted['rating_year'] = df_converted['rating_datetime'].dt.year
    df_converted['rating_month'] = df_converted['rating_datetime'].dt.month
    df_converted['rating_day'] = df_converted['rating_datetime'].dt.day
    df_converted['rating_hour'] = df_converted['rating_datetime'].dt.hour
    df_converted['rating_dayofweek'] = df_converted['rating_datetime'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df_converted['rating_dayname'] = df_converted['rating_datetime'].dt.day_name()
    df_converted['rating_monthname'] = df_converted['rating_datetime'].dt.month_name()
    
    elapsed_time = time.time() - start_time
    
    print(f"   Conversion completada")
    print(f"   Tiempo de ejecucion: {elapsed_time:.4f} segundos")

    print(f"\nESTADISTICAS DE LAS FECHAS:")
    print(f"   Fecha mas antigua: {df_converted['rating_datetime'].min()}")
    print(f"   Fecha mas reciente: {df_converted['rating_datetime'].max()}")
    print(f"   Rango temporal: {(df_converted['rating_datetime'].max() - df_converted['rating_datetime'].min()).days} dias")
    
    print(f"\nMemoria adicional utilizada:")
    memory_before = df.memory_usage(deep=True).sum() / (1024**2)
    memory_after = df_converted.memory_usage(deep=True).sum() / (1024**2)
    print(f"   Antes: {memory_before:.2f} MB")
    print(f"   Despues: {memory_after:.2f} MB")
    print(f"   Incremento: {memory_after - memory_before:.2f} MB")

    print(f"\nANALISIS TEMPORAL:")
    
    print(f"\nDistribucion por ano:")
    year_dist = df_converted['rating_year'].value_counts().sort_index()
    print(f"   Anos con ratings: {year_dist.index.min()} - {year_dist.index.max()}")
    print(f"\nTop 10 anos con mas ratings:")
    for year, count in year_dist.nlargest(10).items():
        print(f"   {year}: {count:,} ratings")
    
    print(f"\nDistribucion por mes:")
    month_dist = df_converted['rating_monthname'].value_counts()
    for month, count in month_dist.items():
        print(f"   {month:10s}: {count:,} ratings")
    
    print(f"\nDistribucion por dia de la semana:")
    day_dist = df_converted['rating_dayname'].value_counts()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in day_order:
        if day in day_dist.index:
            count = day_dist[day]
            print(f"   {day:10s}: {count:,} ratings ({count/len(df_converted)*100:.2f}%)")
    
    print(f"\nDistribucion por hora del dia:")
    hour_dist = df_converted['rating_hour'].value_counts().sort_index()
    print(f"\nTop 10 horas con mas ratings:")
    for hour, count in hour_dist.nlargest(10).items():
        print(f"   {hour:02d}:00 - {hour:02d}:59: {count:,} ratings")

    print(f"\nVista previa con fechas convertidas:")
    preview_cols = ['userId', 'movieId', 'rating', 'timestamp', 'rating_datetime', 'rating_date', 
                    'rating_year', 'rating_month', 'rating_dayname', 'title']
    print(df_converted[preview_cols].head(5).to_string(index=False))
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETADA")
    print("="*60)
    
    return df_converted, elapsed_time


def main():
    """Función principal"""
    print("="*60)
    print("PANDAS - CONVERSION DE TIMESTAMP A FECHA")
    print("="*60)

    df = load_dataset(show_info=False)
    
    if df is None:
        print("ERROR: No se pudo cargar el dataset")
        return None, None

    df_converted, exec_time = convert_timestamp_to_date(df)

    output_file = Path(__file__).parent / "data" / "dataset_with_dates.csv"
    print(f"\nOPCIONAL: Guardar resultado completo?")
    print(f"   Ubicacion: {output_file.name}")
    print(f"   Tamano estimado: ~1.5 GB")
    print(f"   (Descomenta las siguientes lineas en el codigo para guardar)")
    
    sample_file = Path(__file__).parent / "data" / "dataset_with_dates_sample.csv"
    print(f"\nGuardando muestra (primeras 10,000 filas)...")
    df_converted.head(10000).to_csv(sample_file, index=False)
    file_size = sample_file.stat().st_size / (1024**2)
    print(f"   Guardado exitosamente: {sample_file.name}")
    print(f"   Tamano del archivo: {file_size:.2f} MB")
    
    return df_converted, exec_time


if __name__ == "__main__":
    result_df, time_elapsed = main()
