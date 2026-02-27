import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import dask.dataframe as dd
import dask.array as da

# APARTADO 2: CARGA DE DATOS

print("="*50)
print("CARGANDO DATOS DEL DATASET HOUSE PRICES")
print("="*50)

train_data = pd.read_csv('data/train.csv')
print("\n✓ Dataset de entrenamiento cargado correctamente")

test_data = pd.read_csv('data/test.csv')
print("✓ Dataset de prueba cargado correctamente")

# Mostrar información básica del dataset de entrenamiento
print("\n" + "="*50)
print("INFORMACIÓN DEL DATASET DE ENTRENAMIENTO")
print("="*50)
print(f"\nDimensiones: {train_data.shape[0]} filas x {train_data.shape[1]} columnas")
print(f"\nPrimeras 5 filas del dataset:")
print(train_data.head())

print(f"\nInformación general del dataset:")
print(train_data.info())

# Mostrar información básica del dataset de prueba
print("\n" + "="*50)
print("INFORMACIÓN DEL DATASET DE PRUEBA")
print("="*50)
print(f"\nDimensiones: {test_data.shape[0]} filas x {test_data.shape[1]} columnas")
print(f"\nPrimeras 5 filas del dataset:")
print(test_data.head())

print(f"\nInformación general del dataset:")
print(test_data.info())

# Verificar valores nulos
print("\n" + "="*50)
print("VALORES NULOS EN EL DATASET")
print("="*50)
print(f"\nValores nulos en el dataset de entrenamiento:")
print(train_data.isnull().sum()[train_data.isnull().sum() > 0].sort_values(ascending=False))

print(f"\nValores nulos en el dataset de prueba:")
print(test_data.isnull().sum()[test_data.isnull().sum() > 0].sort_values(ascending=False))

# APARTADO 3: ANÁLISIS EXPLORATORIO

#  Estadística descriptiva
print("\n" + "="*50)
print("ESTADÍSTICA DESCRIPTIVA")
print("="*50)

print("\nEstadísticas descriptivas de las variables numéricas:")
print(train_data.describe())

print("\n" + "-"*50)
print("ESTADÍSTICAS ADICIONALES")
print("-"*50)

numerical_cols = train_data.select_dtypes(include=[np.number]).columns

print(f"\nTotal de variables numéricas: {len(numerical_cols)}")
print("\nEstadísticas por variable:")

for col in numerical_cols:
    print(f"\n{col}:")
    print(f"  Media: {train_data[col].mean():.2f}")
    print(f"  Mediana: {train_data[col].median():.2f}")
    print(f"  Desviación estándar: {train_data[col].std():.2f}")
    print(f"  Mínimo: {train_data[col].min():.2f}")
    print(f"  Máximo: {train_data[col].max():.2f}")
    print(f"  Rango: {train_data[col].max() - train_data[col].min():.2f}")
    print(f"  Valores nulos: {train_data[col].isnull().sum()}")

# APARTADO 4: VISUALIZACIÓN

print("\n" + "="*50)
print("VISUALIZACIÓN DE DATOS")
print("="*50)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("\nGenerando histograma de SalePrice...")
plt.figure(figsize=(10, 6))
plt.hist(train_data['SalePrice'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Precio de Venta (SalePrice)')
plt.ylabel('Frecuencia')
plt.title('Distribución del Precio de Venta')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('histograma_saleprice.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Histograma guardado como 'histograma_saleprice.png'")

print("\nGenerando diagramas de dispersión...")

important_vars = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', 'GarageCars', '1stFlrSF']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, var in enumerate(important_vars):
    if var in train_data.columns:
        axes[idx].scatter(train_data[var], train_data['SalePrice'], alpha=0.5)
        axes[idx].set_xlabel(var)
        axes[idx].set_ylabel('SalePrice')
        axes[idx].set_title(f'{var} vs SalePrice')
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_plots_saleprice.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Diagramas de dispersión guardados como 'scatter_plots_saleprice.png'")

print("\nGenerando boxplots...")

categorical_vars = ['OverallQual', 'GarageCars', 'FullBath', 'BedroomAbvGr']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, var in enumerate(categorical_vars):
    if var in train_data.columns:
        train_data.boxplot(column='SalePrice', by=var, ax=axes[idx])
        axes[idx].set_xlabel(var)
        axes[idx].set_ylabel('SalePrice')
        axes[idx].set_title(f'SalePrice por {var}')
        axes[idx].get_figure().suptitle('')

plt.tight_layout()
plt.savefig('boxplots_saleprice.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Boxplots guardados como 'boxplots_saleprice.png'")

print("\nGenerando mapa de calor de correlación...")
plt.figure(figsize=(12, 10))

correlations = train_data[numerical_cols].corr()['SalePrice'].sort_values(ascending=False)

print("\nTop 10 variables con mayor correlación positiva con SalePrice:")
print(correlations.head(11)) 

top_corr_vars = correlations.head(11).index
correlation_matrix = train_data[top_corr_vars].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Mapa de Calor - Variables más correlacionadas con SalePrice')
plt.tight_layout()
plt.savefig('heatmap_correlacion.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Mapa de calor guardado como 'heatmap_correlacion.png'")

print("\n" + "="*50)
print("VISUALIZACIÓN COMPLETADA")
print("="*50)

# APARTADO 5: ANÁLISIS DE CORRELACIÓN

print("\n" + "="*50)
print("ANÁLISIS DE CORRELACIÓN")
print("="*50)

print("\nCalculando matriz de correlación...")
correlation_matrix_full = train_data[numerical_cols].corr()

saleprice_correlations = correlation_matrix_full['SalePrice'].sort_values(ascending=False)

print("\n" + "-"*50)
print("VARIABLES CON MAYOR CORRELACIÓN POSITIVA CON SALEPRICE")
print("-"*50)
print(saleprice_correlations.head(11))  

print("\n" + "-"*50)
print("VARIABLES CON MAYOR CORRELACIÓN NEGATIVA CON SALEPRICE")
print("-"*50)
print(saleprice_correlations.tail(10))

strong_positive_corr = saleprice_correlations[saleprice_correlations > 0.5].drop('SalePrice')
strong_negative_corr = saleprice_correlations[saleprice_correlations < -0.5]

print("\n" + "-"*50)
print("VARIABLES CON CORRELACIÓN FUERTE (|r| > 0.5)")
print("-"*50)
print(f"\nCorrelación positiva fuerte (r > 0.5):")
print(f"Total: {len(strong_positive_corr)} variables")
for var, corr in strong_positive_corr.items():
    print(f"  {var}: {corr:.4f}")

if len(strong_negative_corr) > 0:
    print(f"\nCorrelación negativa fuerte (r < -0.5):")
    print(f"Total: {len(strong_negative_corr)} variables")
    for var, corr in strong_negative_corr.items():
        print(f"  {var}: {corr:.4f}")
else:
    print(f"\nNo hay variables con correlación negativa fuerte (r < -0.5)")

print("\n" + "-"*50)
print("MATRIZ DE CORRELACIÓN COMPLETA")
print("-"*50)
print("\nMatriz de correlación guardada en 'correlation_matrix_full'")
print(f"Dimensiones: {correlation_matrix_full.shape}")

correlation_matrix_full.to_csv('correlation_matrix.csv')
print("\n✓ Matriz de correlación completa guardada en 'correlation_matrix.csv'")

print("\n" + "="*50)
print("ANÁLISIS DE CORRELACIÓN COMPLETADO")
print("="*50)

# APARTADO 6: FILTRADO DE DATOS

print("\n" + "="*50)
print("FILTRADO DE DATOS")
print("="*50)

train_filtered = train_data.copy()
test_filtered = test_data.copy()

print(f"\nDimensiones originales del dataset de entrenamiento: {train_filtered.shape}")

print("\n" + "-"*50)
print("1. ELIMINANDO VARIABLES NO PREDICTIVAS")
print("-"*50)

columns_to_drop = []

if 'Id' in train_filtered.columns:
    columns_to_drop.append('Id')
    print("  - Id: Variable identificadora sin valor predictivo")

# 2. Identificar variables con correlación muy baja con SalePrice
print("\n" + "-"*50)
print("2. IDENTIFICANDO VARIABLES CON BAJA CORRELACIÓN")
print("-"*50)

# Umbral de correlación mínima
correlation_threshold = 0.05

low_correlation_vars = saleprice_correlations[
    (saleprice_correlations.abs() < correlation_threshold) & 
    (saleprice_correlations.index != 'SalePrice')
]

print(f"\nVariables con correlación |r| < {correlation_threshold}:")
print(f"Total: {len(low_correlation_vars)} variables")
for var, corr in low_correlation_vars.items():
    print(f"  - {var}: {corr:.4f}")
    if var not in columns_to_drop:
        columns_to_drop.append(var)

print("\n" + "-"*50)
print("3. IDENTIFICANDO VARIABLES CON EXCESO DE VALORES NULOS")
print("-"*50)

null_threshold = 0.5  
null_percentage = train_filtered.isnull().sum() / len(train_filtered)
high_null_vars = null_percentage[null_percentage > null_threshold]

print(f"\nVariables con más del {null_threshold*100}% de valores nulos:")
print(f"Total: {len(high_null_vars)} variables")
for var, percentage in high_null_vars.items():
    print(f"  - {var}: {percentage*100:.2f}% nulos")
    if var not in columns_to_drop and var != 'SalePrice':
        columns_to_drop.append(var)

print("\n" + "-"*50)
print("RESUMEN DE VARIABLES A ELIMINAR")
print("-"*50)
print(f"\nTotal de variables a eliminar: {len(columns_to_drop)}")
print(f"Variables: {columns_to_drop}")

# 5. Eliminar las columnas identificadas
train_filtered = train_filtered.drop(columns=columns_to_drop, errors='ignore')
test_filtered = test_filtered.drop(columns=columns_to_drop, errors='ignore')

print(f"\nDimensiones después del filtrado:")
print(f"  - Entrenamiento: {train_filtered.shape}")
print(f"  - Prueba: {test_filtered.shape}")

train_filtered.to_csv('data/train_filtered.csv', index=False)
test_filtered.to_csv('data/test_filtered.csv', index=False)

print("\n✓ Datasets filtrados guardados:")
print("  - data/train_filtered.csv")
print("  - data/test_filtered.csv")

print("\n" + "-"*50)
print("INFORMACIÓN DEL DATASET FILTRADO")
print("-"*50)
print(f"\nVariables restantes: {train_filtered.shape[1]}")
print(f"Registros de entrenamiento: {train_filtered.shape[0]}")
print(f"Registros de prueba: {test_filtered.shape[0]}")

print("\nColumnas restantes:")
print(list(train_filtered.columns))

print("\n" + "="*50)
print("FILTRADO DE DATOS COMPLETADO")
print("="*50)

# APARTADO 7: TRATAMIENTO DE VARIABLES CATEGÓRICAS

print("\n" + "="*50)
print("TRATAMIENTO DE VARIABLES CATEGÓRICAS")
print("="*50)

train_encoded = train_filtered.copy()
test_encoded = test_filtered.copy()

print("\n" + "-"*50)
print("1. IDENTIFICANDO VARIABLES CATEGÓRICAS")
print("-"*50)

categorical_cols = train_encoded.select_dtypes(include=['object']).columns.tolist()
print(f"\nTotal de variables categóricas: {len(categorical_cols)}")
print(f"Variables: {categorical_cols}")

y_train = None
if 'SalePrice' in train_encoded.columns:
    y_train = train_encoded['SalePrice'].copy()
    train_encoded = train_encoded.drop('SalePrice', axis=1)
    print(f"\n✓ Variable objetivo (SalePrice) separada")

print("\n" + "-"*50)
print("2. APLICANDO LABEL ENCODING (VARIABLES ORDINALES)")
print("-"*50)

ordinal_mappings = {
    'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
}

ordinal_encoded = []
for col, mapping in ordinal_mappings.items():
    if col in train_encoded.columns:
        # Rellenar valores nulos con 'NA' antes de mapear
        train_encoded[col] = train_encoded[col].fillna('NA').map(mapping)
        test_encoded[col] = test_encoded[col].fillna('NA').map(mapping)
        ordinal_encoded.append(col)
        print(f"  ✓ {col}: Mapeado a valores ordinales")

print(f"\nTotal de variables con label encoding: {len(ordinal_encoded)}")

print("\n" + "-"*50)
print("3. APLICANDO ONE-HOT ENCODING (VARIABLES NOMINALES)")
print("-"*50)

categorical_cols = train_encoded.select_dtypes(include=['object']).columns.tolist()

print(f"\nVariables categóricas restantes: {len(categorical_cols)}")

if len(categorical_cols) > 0:

    train_encoded = pd.get_dummies(train_encoded, columns=categorical_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_encoded, columns=categorical_cols, drop_first=True)
    
    print(f"\n✓ One-Hot Encoding aplicado a {len(categorical_cols)} variables")
    
    missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
    for col in missing_cols:
        test_encoded[col] = 0
    
    extra_cols = set(test_encoded.columns) - set(train_encoded.columns)
    test_encoded = test_encoded.drop(columns=extra_cols)
    
    test_encoded = test_encoded[train_encoded.columns]
    
    print(f"✓ Columnas alineadas entre train y test")

if y_train is not None:
    train_encoded['SalePrice'] = y_train
    print(f"\n✓ Variable objetivo (SalePrice) agregada de nuevo al dataset")

train_encoded.to_csv('data/train_encoded.csv', index=False)
test_encoded.to_csv('data/test_encoded.csv', index=False)

print("\n" + "-"*50)
print("RESUMEN DEL ENCODING")
print("-"*50)
print(f"\nDimensiones después del encoding:")
print(f"  - Entrenamiento: {train_encoded.shape}")
print(f"  - Prueba: {test_encoded.shape}")

print("\n✓ Datasets codificados guardados:")
print("  - data/train_encoded.csv")
print("  - data/test_encoded.csv")

print("\n" + "="*50)
print("TRATAMIENTO DE VARIABLES CATEGÓRICAS COMPLETADO")
print("="*50)

# APARTADO 8: TRATAMIENTO DE VALORES FALTANTES

print("\n" + "="*50)
print("TRATAMIENTO DE VALORES FALTANTES")
print("="*50)

train_complete = train_encoded.copy()
test_complete = test_encoded.copy()

print("\n" + "-"*50)
print("1. IDENTIFICANDO VALORES FALTANTES")
print("-"*50)

y_train_complete = None
if 'SalePrice' in train_complete.columns:
    y_train_complete = train_complete['SalePrice'].copy()
    train_complete = train_complete.drop('SalePrice', axis=1)

train_nulls = train_complete.isnull().sum()
test_nulls = test_complete.isnull().sum()

train_nulls_cnt = train_nulls[train_nulls > 0].sort_values(ascending=False)
test_nulls_cnt = test_nulls[test_nulls > 0].sort_values(ascending=False)

print(f"\nVariables con valores faltantes en TRAIN: {len(train_nulls_cnt)}")
if len(train_nulls_cnt) > 0:
    for col, count in train_nulls_cnt.items():
        percentage = (count / len(train_complete)) * 100
        print(f"  - {col}: {count} ({percentage:.2f}%)")

print(f"\nVariables con valores faltantes en TEST: {len(test_nulls_cnt)}")
if len(test_nulls_cnt) > 0:
    for col, count in test_nulls_cnt.items():
        percentage = (count / len(test_complete)) * 100
        print(f"  - {col}: {count} ({percentage:.2f}%)")

print("\n" + "-"*50)
print("2. APLICANDO ESTRATEGIAS DE IMPUTACIÓN")
print("-"*50)

numerical_features = train_complete.select_dtypes(include=[np.number]).columns

print(f"\nTotal de variables numéricas: {len(numerical_features)}")

print("\nImputando valores numéricos con la MEDIANA...")
for col in numerical_features:
    if train_complete[col].isnull().sum() > 0 or test_complete[col].isnull().sum() > 0:
        median_value = train_complete[col].median()
        
        train_nulls_before = train_complete[col].isnull().sum()
        test_nulls_before = test_complete[col].isnull().sum()
        
        train_complete[col] = train_complete[col].fillna(median_value)
        test_complete[col] = test_complete[col].fillna(median_value)
        
        if train_nulls_before > 0 or test_nulls_before > 0:
            print(f"  ✓ {col}: Imputados {train_nulls_before} (train) y {test_nulls_before} (test) con mediana = {median_value:.2f}")

print("\n" + "-"*50)
print("3. VERIFICACIÓN FINAL")
print("-"*50)

train_remaining_nulls = train_complete.isnull().sum().sum()
test_remaining_nulls = test_complete.isnull().sum().sum()

print(f"\nValores nulos restantes en TRAIN: {train_remaining_nulls}")
print(f"Valores nulos restantes en TEST: {test_remaining_nulls}")

if train_remaining_nulls == 0 and test_remaining_nulls == 0:
    print("\n✓ Todos los valores faltantes han sido tratados correctamente")
else:
    print("\n⚠ Aún quedan valores faltantes por tratar")

if y_train_complete is not None:
    train_complete['SalePrice'] = y_train_complete
    print(f"\n✓ Variable objetivo (SalePrice) agregada de nuevo al dataset")

train_complete.to_csv('data/train_complete.csv', index=False)
test_complete.to_csv('data/test_complete.csv', index=False)

print("\n" + "-"*50)
print("RESUMEN DEL TRATAMIENTO")
print("-"*50)
print(f"\nDimensiones finales:")
print(f"  - Entrenamiento: {train_complete.shape}")
print(f"  - Prueba: {test_complete.shape}")

print("\n✓ Datasets completos guardados:")
print("  - data/train_complete.csv")
print("  - data/test_complete.csv")

print("\n" + "="*50)
print("TRATAMIENTO DE VALORES FALTANTES COMPLETADO")
print("="*50)

# APARTADO 9: TRANSFORMACIONES

print("\n" + "="*50)
print("TRANSFORMACIONES")
print("="*50)

train_transformed = train_complete.copy()
test_transformed = test_complete.copy()

print("\n" + "-"*50)
print("1. TRANSFORMACIÓN LOGARÍTMICA DE SALEPRICE")
print("-"*50)

if 'SalePrice' in train_transformed.columns:

    skewness_before = train_transformed['SalePrice'].skew()
    print(f"\nAsimetría de SalePrice (antes): {skewness_before:.4f}")
    
    train_transformed['SalePrice_Log'] = np.log1p(train_transformed['SalePrice'])
    
    skewness_after = train_transformed['SalePrice_Log'].skew()
    print(f"Asimetría de SalePrice_Log (después): {skewness_after:.4f}")
    print(f"\n✓ Transformación logarítmica aplicada a SalePrice")
    print(f"  Mejora en asimetría: {abs(skewness_before) - abs(skewness_after):.4f}")

print("\n" + "-"*50)
print("2. IDENTIFICANDO VARIABLES CON ALTA ASIMETRÍA")
print("-"*50)

y_train_transformed = None
if 'SalePrice' in train_transformed.columns:
    y_train_transformed = train_transformed['SalePrice'].copy()
    saleprice_log = train_transformed['SalePrice_Log'].copy()
    train_transformed = train_transformed.drop(['SalePrice', 'SalePrice_Log'], axis=1)

numerical_features = train_transformed.select_dtypes(include=[np.number]).columns

skewness = train_transformed[numerical_features].skew().sort_values(ascending=False)

skewness_threshold = 0.75
high_skew_features = skewness[abs(skewness) > skewness_threshold]

print(f"\nVariables con asimetría |skew| > {skewness_threshold}: {len(high_skew_features)}")
print("\nTop 10 variables más asimétricas:")
for col, skew_val in high_skew_features.head(10).items():
    print(f"  - {col}: {skew_val:.4f}")

print("\n" + "-"*50)
print("3. APLICANDO TRANSFORMACIÓN LOGARÍTMICA")
print("-"*50)

transformed_count = 0
for col in high_skew_features.index:
    if col in train_transformed.columns:
        skew_before = train_transformed[col].skew()
        
        train_transformed[col] = np.log1p(train_transformed[col] - train_transformed[col].min() + 1)
        test_transformed[col] = np.log1p(test_transformed[col] - test_transformed[col].min() + 1)
        
        skew_after = train_transformed[col].skew()
        transformed_count += 1
        
        if transformed_count <= 5:  
            print(f"  ✓ {col}: skew {skew_before:.4f} → {skew_after:.4f}")

print(f"\n✓ Total de variables transformadas: {transformed_count}")

print("\n" + "-"*50)
print("4. ESCALADO DE VARIABLES (STANDARDIZATION)")
print("-"*50)

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_transformed)
test_scaled = scaler.transform(test_transformed)

train_transformed = pd.DataFrame(train_scaled, columns=train_transformed.columns)
test_transformed = pd.DataFrame(test_scaled, columns=test_transformed.columns)

print("\n✓ Escalado StandardScaler aplicado")
print(f"  Media ≈ 0, Desviación estándar ≈ 1")

print(f"\nVerificación del escalado (train):")
print(f"  Media de las variables: {train_transformed.mean().mean():.6f}")
print(f"  Desviación estándar promedio: {train_transformed.std().mean():.6f}")

if y_train_transformed is not None:
    train_transformed['SalePrice'] = y_train_transformed.values
    train_transformed['SalePrice_Log'] = saleprice_log.values
    print(f"\n✓ Variables objetivo agregadas de nuevo al dataset")

train_transformed.to_csv('data/train_transformed.csv', index=False)
test_transformed.to_csv('data/test_transformed.csv', index=False)

print("\n" + "-"*50)
print("RESUMEN DE TRANSFORMACIONES")
print("-"*50)
print(f"\nDimensiones finales:")
print(f"  - Entrenamiento: {train_transformed.shape}")
print(f"  - Prueba: {test_transformed.shape}")

print(f"\nTransformaciones aplicadas:")
print(f"  - Transformación logarítmica: {transformed_count + 1} variables (incluyendo SalePrice)")
print(f"  - Escalado StandardScaler: Todas las variables numéricas")

print("\n✓ Datasets transformados guardados:")
print("  - data/train_transformed.csv")
print("  - data/test_transformed.csv")

print("\n" + "="*50)
print("TRANSFORMACIONES COMPLETADAS")
print("="*50)

