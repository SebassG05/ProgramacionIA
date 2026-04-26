import pandas as pd
import numpy as np
import time
from pathlib import Path
from load_data import load_dataset


def calculate_user_statistics(df):
    """
    Calcula media y desviación estándar de ratings por usuario
    
    Args:
        df (pd.DataFrame): Dataset completo
        
    Returns:
        pd.DataFrame: Dataset con estadísticas por usuario
        float: Tiempo de ejecución
    """
    print("\n" + "="*60)
    print("CÁLCULO DE ESTADÍSTICAS POR USUARIO")
    print("="*60)
    
    print(f"\nDataset original:")
    print(f"   Total de ratings: {len(df):,}")
    print(f"   Usuarios únicos: {df['userId'].nunique():,}")
    
    # Medir tiempo de cálculo
    start_time = time.time()
    
    # Calcular estadísticas por usuario
    print(f"\nCalculando media y desviación estándar por usuario...")
    user_stats = df.groupby('userId')['rating'].agg([
        ('mean_rating', 'mean'),
        ('std_rating', 'std'),
        ('count_ratings', 'count'),
        ('min_rating', 'min'),
        ('max_rating', 'max')
    ]).reset_index()
    
    # Manejar usuarios con solo 1 rating (std = NaN)
    user_stats['std_rating'] = user_stats['std_rating'].fillna(0)
    
    elapsed_time = time.time() - start_time
    
    print(f"   Estadísticas calculadas exitosamente")
    print(f"   Tiempo de ejecución: {elapsed_time:.4f} segundos")
    
    # Estadísticas generales
    print(f"\nESTADÍSTICAS GENERALES:")
    print(f"   Total de usuarios: {len(user_stats):,}")
    print(f"   Memoria utilizada: {user_stats.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    print(f"\nDistribución de medias de rating:")
    print(f"   Media global: {user_stats['mean_rating'].mean():.3f}")
    print(f"   Mediana: {user_stats['mean_rating'].median():.3f}")
    print(f"   Desviación estándar: {user_stats['mean_rating'].std():.3f}")
    print(f"   Mínimo: {user_stats['mean_rating'].min():.3f}")
    print(f"   Máximo: {user_stats['mean_rating'].max():.3f}")
    
    print(f"\nDistribución de desviaciones estándar:")
    print(f"   Media de std: {user_stats['std_rating'].mean():.3f}")
    print(f"   Mediana de std: {user_stats['std_rating'].median():.3f}")
    print(f"   Usuarios con std = 0 (solo 1 rating o todos iguales): {(user_stats['std_rating'] == 0).sum():,}")
    
    print(f"\nDistribución de cantidad de ratings:")
    print(f"   Promedio de ratings por usuario: {user_stats['count_ratings'].mean():.1f}")
    print(f"   Mediana: {user_stats['count_ratings'].median():.0f}")
    print(f"   Usuario más activo: {user_stats['count_ratings'].max():,} ratings")
    print(f"   Usuario menos activo: {user_stats['count_ratings'].min():,} ratings")
    
    # Top usuarios más activos
    print(f"\nTop 10 usuarios más activos:")
    top_users = user_stats.nlargest(10, 'count_ratings')
    for idx, row in top_users.iterrows():
        print(f"   User {int(row['userId']):6d}: {int(row['count_ratings']):6,} ratings | "
              f"Media: {row['mean_rating']:.2f} | Std: {row['std_rating']:.2f}")
    
    # Usuarios más críticos (media más baja con suficientes ratings)
    print(f"\nTop 10 usuarios más críticos (mínimo 100 ratings):")
    critical_users = user_stats[user_stats['count_ratings'] >= 100].nsmallest(10, 'mean_rating')
    for idx, row in critical_users.iterrows():
        print(f"   User {int(row['userId']):6d}: Media: {row['mean_rating']:.2f} | "
              f"{int(row['count_ratings']):,} ratings | Std: {row['std_rating']:.2f}")
    
    # Usuarios más generosos (media más alta con suficientes ratings)
    print(f"\nTop 10 usuarios más generosos (mínimo 100 ratings):")
    generous_users = user_stats[user_stats['count_ratings'] >= 100].nlargest(10, 'mean_rating')
    for idx, row in generous_users.iterrows():
        print(f"   User {int(row['userId']):6d}: Media: {row['mean_rating']:.2f} | "
              f"{int(row['count_ratings']):,} ratings | Std: {row['std_rating']:.2f}")
    
    print(f"\nTop 10 usuarios más consistentes (std más baja, mínimo 50 ratings):")
    consistent_users = user_stats[user_stats['count_ratings'] >= 50].nsmallest(10, 'std_rating')
    for idx, row in consistent_users.iterrows():
        print(f"   User {int(row['userId']):6d}: Std: {row['std_rating']:.2f} | "
              f"Media: {row['mean_rating']:.2f} | {int(row['count_ratings']):,} ratings")
    
    print(f"\nTop 10 usuarios más variables (std más alta, mínimo 50 ratings):")
    variable_users = user_stats[user_stats['count_ratings'] >= 50].nlargest(10, 'std_rating')
    for idx, row in variable_users.iterrows():
        print(f"   User {int(row['userId']):6d}: Std: {row['std_rating']:.2f} | "
              f"Media: {row['mean_rating']:.2f} | {int(row['count_ratings']):,} ratings")
    
    print(f"\nPrimeras 10 filas del resultado:")
    print(user_stats.head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("CALCULO COMPLETADO")
    print("="*60)
    
    return user_stats, elapsed_time


def main():
    """Función principal"""
    print("="*60)
    print("PANDAS - ESTADÍSTICAS DE RATINGS POR USUARIO")
    print("="*60)
    
    df = load_dataset(show_info=False)
    
    if df is None:
        print("ERROR: No se pudo cargar el dataset")
        return None, None
    
    user_stats, exec_time = calculate_user_statistics(df)
    
    output_file = Path(__file__).parent / "data" / "user_statistics.csv"
    print(f"\nGuardando resultado en: {output_file.name}")
    user_stats.to_csv(output_file, index=False)
    file_size = output_file.stat().st_size / (1024**2)
    print(f"   Guardado exitosamente")
    print(f"   Tamaño del archivo: {file_size:.2f} MB")
    
    return user_stats, exec_time


if __name__ == "__main__":
    result_df, time_elapsed = main()
