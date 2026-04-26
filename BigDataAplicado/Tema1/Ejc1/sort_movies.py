import pandas as pd
import time
from pathlib import Path
from load_data import load_dataset


def sort_movies_by_rating(df, min_ratings=50):
    """
    Ordena películas por rating promedio
    
    Args:
        df (pd.DataFrame): Dataset completo
        min_ratings (int): Mínimo de ratings para considerar una película (default: 50)
        
    Returns:
        pd.DataFrame: Dataset con películas ordenadas por rating promedio
        float: Tiempo de ejecución
    """
    print("\n" + "="*60)
    print("ORDENAR PELICULAS POR RATING PROMEDIO")
    print("="*60)
    
    print(f"\nDataset original:")
    print(f"   Total de ratings: {len(df):,}")
    print(f"   Peliculas unicas: {df['movieId'].nunique():,}")
    
    start_time = time.time()
    
    print(f"\nCalculando estadisticas por pelicula...")
    movie_stats = df.groupby(['movieId', 'title', 'genres'])['rating'].agg([
        ('mean_rating', 'mean'),
        ('std_rating', 'std'),
        ('count_ratings', 'count'),
        ('min_rating', 'min'),
        ('max_rating', 'max')
    ]).reset_index()
    
    movie_stats['std_rating'] = movie_stats['std_rating'].fillna(0)
    
    print(f"   Estadisticas calculadas exitosamente")
    print(f"   Total de peliculas: {len(movie_stats):,}")
    
    print(f"\nFiltrando peliculas con minimo {min_ratings} ratings...")
    filtered_movies = movie_stats[movie_stats['count_ratings'] >= min_ratings].copy()
    print(f"   Peliculas filtradas: {len(filtered_movies):,}")
    print(f"   Reduccion: {((1 - len(filtered_movies)/len(movie_stats)) * 100):.2f}%")
    
    print(f"\nOrdenando peliculas por rating promedio...")
    sorted_movies = filtered_movies.sort_values('mean_rating', ascending=False).reset_index(drop=True)
    
    elapsed_time = time.time() - start_time
    
    print(f"   Ordenamiento completado")
    print(f"   Tiempo de ejecucion: {elapsed_time:.4f} segundos")
    
    print(f"\nESTADISTICAS DEL RESULTADO:")
    print(f"   Memoria utilizada: {sorted_movies.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    print(f"\nDistribucion de ratings promedio:")
    print(f"   Media global: {sorted_movies['mean_rating'].mean():.3f}")
    print(f"   Mediana: {sorted_movies['mean_rating'].median():.3f}")
    print(f"   Desviacion estandar: {sorted_movies['mean_rating'].std():.3f}")
    print(f"   Minimo: {sorted_movies['mean_rating'].min():.3f}")
    print(f"   Maximo: {sorted_movies['mean_rating'].max():.3f}")
    
    print(f"\nDistribucion de cantidad de ratings:")
    print(f"   Promedio de ratings por pelicula: {sorted_movies['count_ratings'].mean():.1f}")
    print(f"   Mediana: {sorted_movies['count_ratings'].median():.0f}")
    print(f"   Pelicula mas valorada: {sorted_movies['count_ratings'].max():,} ratings")
    print(f"   Pelicula menos valorada (del filtro): {sorted_movies['count_ratings'].min():,} ratings")
    
    print(f"\nTop 20 peliculas mejor valoradas (minimo {min_ratings} ratings):")
    top_movies = sorted_movies.head(20)
    for idx, row in top_movies.iterrows():
        print(f"   {idx+1:2d}. [{row['mean_rating']:.3f}] {row['title']}")
        print(f"       Ratings: {int(row['count_ratings']):,} | Std: {row['std_rating']:.2f} | Generos: {row['genres']}")

    print(f"\nBottom 10 peliculas peor valoradas (minimo {min_ratings} ratings):")
    bottom_movies = sorted_movies.tail(10)
    for idx, row in bottom_movies.iterrows():
        print(f"   {len(sorted_movies) - idx}. [{row['mean_rating']:.3f}] {row['title']}")
        print(f"       Ratings: {int(row['count_ratings']):,} | Std: {row['std_rating']:.2f} | Generos: {row['genres']}")
    
    print(f"\nTop 10 peliculas con mas ratings:")
    most_rated = sorted_movies.nlargest(10, 'count_ratings')
    for idx, row in most_rated.iterrows():
        print(f"   {int(row['count_ratings']):6,} ratings: [{row['mean_rating']:.3f}] {row['title']}")

    print(f"\nAnalisis por genero principal:")
    sorted_movies['main_genre'] = sorted_movies['genres'].str.split('|').str[0]
    genre_stats = sorted_movies.groupby('main_genre').agg({
        'mean_rating': 'mean',
        'count_ratings': 'sum',
        'movieId': 'count'
    }).sort_values('mean_rating', ascending=False).head(10)
    genre_stats.columns = ['Avg_Rating', 'Total_Ratings', 'Num_Movies']
    
    print("\nTop 10 generos por rating promedio:")
    for genre, row in genre_stats.iterrows():
        print(f"   {genre:20s}: Rating: {row['Avg_Rating']:.3f} | Peliculas: {int(row['Num_Movies']):4,} | Total ratings: {int(row['Total_Ratings']):8,}")
    
    # Vista previa
    print(f"\nPrimeras 10 peliculas del ranking:")
    print(sorted_movies[['title', 'mean_rating', 'count_ratings', 'std_rating', 'genres']].head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("ORDENAMIENTO COMPLETADO")
    print("="*60)
    
    return sorted_movies, elapsed_time


def main():
    """Función principal"""
    print("="*60)
    print("PANDAS - ORDENAR PELICULAS POR RATING PROMEDIO")
    print("="*60)
    
    # Cargar dataset
    df = load_dataset(show_info=False)
    
    if df is None:
        print("ERROR: No se pudo cargar el dataset")
        return None, None

    sorted_movies, exec_time = sort_movies_by_rating(df, min_ratings=50)
    
    output_file = Path(__file__).parent / "data" / "movies_sorted_by_rating.csv"
    print(f"\nGuardando resultado en: {output_file.name}")
    sorted_movies.to_csv(output_file, index=False)
    file_size = output_file.stat().st_size / (1024**2)
    print(f"   Guardado exitosamente")
    print(f"   Tamano del archivo: {file_size:.2f} MB")
    
    return sorted_movies, exec_time


if __name__ == "__main__":
    result_df, time_elapsed = main()
