import pandas as pd
from pathlib import Path
import time


# Configuración de rutas
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "ml-20m" / "ml-20m"
RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"
OUTPUT_FILE = BASE_DIR / "data" / "movielens_superdataset.csv"


def create_superdataset():
    """
    Combina ratings.csv y movies.csv en un único superdataset
    con los campos: userId, movieId, rating, timestamp, title, genres
    """
    print("="*60)
    print("CREACIÓN DE SUPERDATASET MOVIELENS 20M")
    print("="*60 + "\n")
    
    # Verificar archivos de entrada
    if not RATINGS_FILE.exists():
        print(f"❌ ERROR: No se encuentra {RATINGS_FILE}")
        return False
    
    if not MOVIES_FILE.exists():
        print(f"❌ ERROR: No se encuentra {MOVIES_FILE}")
        return False
    
    start_time = time.time()
    
    # Cargar ratings
    print(f"📂 Cargando ratings.csv...")
    print(f"   Tamaño: {RATINGS_FILE.stat().st_size / (1024**2):.2f} MB")
    ratings = pd.read_csv(RATINGS_FILE)
    print(f"   ✓ Cargado: {len(ratings):,} filas")
    print(f"   Columnas: {', '.join(ratings.columns.tolist())}")
    
    # Cargar movies
    print(f"\n📂 Cargando movies.csv...")
    print(f"   Tamaño: {MOVIES_FILE.stat().st_size / (1024**2):.2f} MB")
    movies = pd.read_csv(MOVIES_FILE)
    print(f"   ✓ Cargado: {len(movies):,} filas")
    print(f"   Columnas: {', '.join(movies.columns.tolist())}")
    
    # Merge de datasets
    print(f"\n🔗 Combinando datasets...")
    superdataset = ratings.merge(movies, on='movieId', how='left')
    
    print(f"   ✓ Combinación exitosa!")
    print(f"   Filas finales: {len(superdataset):,}")
    print(f"   Columnas finales: {', '.join(superdataset.columns.tolist())}")
    
    # Información del resultado
    print(f"\n📊 Información del superdataset:")
    print(f"   Memoria: {superdataset.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"   Valores nulos:")
    null_counts = superdataset.isnull().sum()
    if null_counts.sum() > 0:
        for col, count in null_counts[null_counts > 0].items():
            print(f"      {col}: {count:,}")
    else:
        print(f"      No hay valores nulos ✓")
    
    # Vista previa
    print(f"\n🔍 Primeras 3 filas:")
    print(superdataset.head(3).to_string())
    
    # Guardar superdataset
    print(f"\n💾 Guardando superdataset en: {OUTPUT_FILE.name}")
    superdataset.to_csv(OUTPUT_FILE, index=False)
    
    file_size = OUTPUT_FILE.stat().st_size / (1024**2)
    elapsed_time = time.time() - start_time
    
    print(f"   ✓ Guardado exitosamente!")
    print(f"   Tamaño del archivo: {file_size:.2f} MB")
    print(f"   Tiempo total: {elapsed_time:.2f} segundos")
    
    print("\n" + "="*60)
    print("✅ SUPERDATASET CREADO EXITOSAMENTE")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = create_superdataset()
    if not success:
        print("\n❌ Error al crear el superdataset")
        exit(1)
