import pandas as pd
import time
from pathlib import Path
from load_data import load_dataset


def filter_action_movies(df, min_rating=4.0):
    """
    Filtra películas de género Action con rating >= min_rating
    
    Args:
        df (pd.DataFrame): Dataset completo
        min_rating (float): Rating mínimo (default: 4.0)
        
    Returns:
        pd.DataFrame: Dataset filtrado
    """
    print("\n" + "="*60)
    print("FILTRADO DE PELÍCULAS DE ACCIÓN")
    print("="*60)
    
    print(f"\n📊 Dataset original:")
    print(f"   Total de filas: {len(df):,}")
    print(f"   Columnas: {', '.join(df.columns.tolist())}")
    
    start_time = time.time()
    
    print(f"\n🎬 Filtrando por género 'Action'...")
    action_mask = df['genres'].str.contains('Action', case=False, na=False)
    action_df = df[action_mask]
    print(f"   ✓ Películas con género Action: {len(action_df):,}")
    
    print(f"\n⭐ Filtrando por rating >= {min_rating}...")
    rating_mask = action_df['rating'] >= min_rating
    filtered_df = action_df[rating_mask]
    print(f"   ✓ Películas con rating >= {min_rating}: {len(filtered_df):,}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n📈 ESTADÍSTICAS DEL FILTRADO:")
    print(f"   Tiempo de ejecución: {elapsed_time:.4f} segundos")
    print(f"   Filas originales: {len(df):,}")
    print(f"   Filas filtradas: {len(filtered_df):,}")
    print(f"   Reducción: {((1 - len(filtered_df)/len(df)) * 100):.2f}%")
    print(f"   Rating promedio: {filtered_df['rating'].mean():.3f}")
    print(f"   Rating mediano: {filtered_df['rating'].median():.3f}")
    print(f"   Rating mínimo: {filtered_df['rating'].min():.1f}")
    print(f"   Rating máximo: {filtered_df['rating'].max():.1f}")
    print(f"   Usuarios únicos: {filtered_df['userId'].nunique():,}")
    print(f"   Películas únicas: {filtered_df['movieId'].nunique():,}")

    print(f"\n🎭 Top 10 combinaciones de géneros más frecuentes:")
    top_genres = filtered_df['genres'].value_counts().head(10)
    for idx, (genre, count) in enumerate(top_genres.items(), 1):
        print(f"   {idx:2d}. {genre}: {count:,} ratings")
    
    print(f"\n🏆 Top 5 películas de Action más valoradas (por cantidad):")
    top_movies = filtered_df.groupby(['movieId', 'title']).size().sort_values(ascending=False).head(5)
    for idx, ((movie_id, title), count) in enumerate(top_movies.items(), 1):
        avg_rating = filtered_df[filtered_df['movieId'] == movie_id]['rating'].mean()
        print(f"   {idx}. {title}")
        print(f"      Ratings: {count:,} | Rating promedio: {avg_rating:.2f}")
    
    print(f"\n🔍 Primeras 5 filas del resultado:")
    print(filtered_df.head().to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ FILTRADO COMPLETADO")
    print("="*60)
    
    return filtered_df, elapsed_time


def main():
    """Función principal"""
    print("="*60)
    print("PANDAS - FILTRADO DE PELÍCULAS DE ACCIÓN")
    print("="*60)
    
    df = load_dataset(show_info=False)
    
    if df is None:
        print("❌ Error al cargar el dataset")
        return None, None
    
    filtered_df, exec_time = filter_action_movies(df, min_rating=4.0)

    output_file = Path(__file__).parent / "data" / "action_movies_filtered.csv"
    print(f"\n💾 ¿Deseas guardar el resultado filtrado?")
    print(f"   Ubicación: {output_file}")
    print(f"   Tamaño estimado: {len(filtered_df) * 100 / 1024 / 1024:.2f} MB")
    # Descomentar para guardar:
    # filtered_df.to_csv(output_file, index=False)
    # print(f"   ✓ Guardado exitosamente")
    
    return filtered_df, exec_time


if __name__ == "__main__":
    result_df, time_elapsed = main()
