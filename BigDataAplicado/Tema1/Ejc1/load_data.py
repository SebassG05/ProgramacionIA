import pandas as pd
import os
from pathlib import Path
import sys


# Configuración de rutas
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATASET_FILE = DATA_DIR / "movielens_superdataset.csv"


def check_dataset_exists():
    """Verifica si el dataset existe en la ruta especificada"""
    if not DATASET_FILE.exists():
        print(f"❌ ERROR: No se encontró el archivo {DATASET_FILE}")
        print(f"\n📍 Ubicación esperada: {DATASET_FILE.absolute()}")
        print("\n💡 Instrucciones:")
        print("   1. Descarga el dataset MovieLens 20M")
        print("   2. Coloca el archivo 'movielens_superdataset.csv' en la carpeta 'data/'")
        return False
    return True


def load_dataset(show_info=True):
    """
    Carga el dataset MovieLens con manejo de errores
    
    Args:
        show_info (bool): Si True, muestra información del dataset
        
    Returns:
        pd.DataFrame: Dataset cargado o None si hay error
    """
    try:
        # Verificar existencia del archivo
        if not check_dataset_exists():
            return None
        
        print(f"📂 Cargando dataset desde: {DATASET_FILE.name}")
        print(f"📊 Tamaño del archivo: {DATASET_FILE.stat().st_size / (1024**2):.2f} MB")
        
        # Cargar dataset
        df = pd.read_csv(DATASET_FILE)
        
        print(f"✅ Dataset cargado exitosamente!")
        
        if show_info:
            print("\n" + "="*50)
            print("📋 INFORMACIÓN DEL DATASET")
            print("="*50)
            print(f"Número de filas: {len(df):,}")
            print(f"Número de columnas: {len(df.columns)}")
            print(f"\nColumnas: {', '.join(df.columns.tolist())}")
            print(f"\nTipos de datos:")
            print(df.dtypes)
            print(f"\nMemoria utilizada: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            print(f"\n🔍 Primeras 5 filas:")
            print(df.head())
            print(f"\n📊 Información de valores nulos:")
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                print(null_counts[null_counts > 0])
            else:
                print("No hay valores nulos ✓")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo no encontrado: {DATASET_FILE}")
        return None
    except pd.errors.EmptyDataError:
        print(f"❌ ERROR: El archivo está vacío")
        return None
    except pd.errors.ParserError as e:
        print(f"❌ ERROR al parsear el CSV: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR inesperado: {type(e).__name__}: {e}")
        return None


def main():
    """Función principal"""
    print("="*50)
    print("CARGA DE DATASET MOVIELENS 20M")
    print("="*50 + "\n")
    
    # Crear directorio data si no existe
    DATA_DIR.mkdir(exist_ok=True)
    print(f"✓ Directorio de datos: {DATA_DIR.absolute()}\n")
    
    # Cargar dataset
    df = load_dataset(show_info=True)
    
    if df is not None:
        print("\n" + "="*50)
        print("✅ CARGA COMPLETADA")
        print("="*50)
        return df
    else:
        print("\n" + "="*50)
        print("❌ CARGA FALLIDA")
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    dataset = main()
