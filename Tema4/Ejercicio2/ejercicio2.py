import pandas as pd
import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV

# APARTADO 1: MODELADO Y EVALUACIÓN

print("="*50)
print("MODELADO Y EVALUACIÓN - EJERCICIO 2")
print("="*50)

print("\n" + "-"*50)
print("1. CARGANDO DATOS PREPROCESADOS")
print("-"*50)

train_data = pd.read_csv('../Ejercicio1/data/train_transformed.csv')
test_data = pd.read_csv('../Ejercicio1/data/test_transformed.csv')

print(f"\n✓ Datos de entrenamiento cargados: {train_data.shape}")
print(f"✓ Datos de prueba cargados: {test_data.shape}")

if 'SalePrice_Log' in train_data.columns:
    y_train = train_data['SalePrice_Log'].copy()
    X_train = train_data.drop(['SalePrice', 'SalePrice_Log'], axis=1)
    print(f"\n✓ Variable objetivo: SalePrice_Log")
elif 'SalePrice' in train_data.columns:
    y_train = train_data['SalePrice'].copy()
    X_train = train_data.drop(['SalePrice'], axis=1)
    print(f"\n✓ Variable objetivo: SalePrice")

X_test = test_data.copy()

print(f"\nDimensiones de los conjuntos:")
print(f"  - X_train: {X_train.shape}")
print(f"  - y_train: {y_train.shape}")
print(f"  - X_test: {X_test.shape}")

print("\n" + "-"*50)
print("2. IMPLEMENTANDO MODELO DE REGRESIÓN")
print("-"*50)

print("\nModelo elegido: Ridge Regression (Regresión con regularización L2)")
print("\nJustificación:")
print("  - El dataset tiene muchas características después del preprocesamiento")
print("  - Ridge es robusto ante multicolinealidad")
print("  - La regularización L2 previene overfitting")
print("  - Adecuado para problemas de regresión con múltiples variables")

model = Ridge(
    alpha=1.0,
    fit_intercept=True,
    max_iter=1000,
    random_state=42
)

print("\n" + "-"*50)
print("HIPERPARÁMETROS INICIALIZADOS")
print("-"*50)
print(f"\n  - alpha: 1.0 (parámetro de regularización)")
print(f"  - fit_intercept: True (calcular término independiente)")
print(f"  - max_iter: 1000 (máximo de iteraciones)")
print(f"  - random_state: 42 (semilla para reproducibilidad)")

print("\n" + "="*50)
print("MODELO INICIALIZADO CORRECTAMENTE")
print("="*50)

# 2. ENTRENAMIENTO DEL MODELO

print("\n" + "-"*50)
print("3. ENTRENANDO EL MODELO")
print("-"*50)

print("\nIniciando entrenamiento del modelo Ridge Regression...")
print(f"  Datos de entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} características")

model.fit(X_train, y_train)

print("\n✓ Modelo entrenado exitosamente")

print("\n" + "-"*50)
print("INFORMACIÓN DEL MODELO ENTRENADO")
print("-"*50)
print(f"\n  - Coeficientes del modelo: {len(model.coef_)} coeficientes")
print(f"  - Intercepto (término independiente): {model.intercept_:.4f}")
print(f"  - Número de iteraciones: {model.n_iter_}")

feature_importance = pd.DataFrame({
    'Variable': X_train.columns,
    'Coeficiente': np.abs(model.coef_)
}).sort_values('Coeficiente', ascending=False)

print(f"\nTop 10 variables más importantes (por magnitud del coeficiente):")
print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*50)
print("ENTRENAMIENTO COMPLETADO")
print("="*50)

# 4. EVALUACIÓN DEL MODELO

print("\n" + "-"*50)
print("4. EVALUANDO EL MODELO")
print("-"*50)

print("\nRealizando predicciones sobre el conjunto de entrenamiento...")

y_train_pred = model.predict(X_train)

print("✓ Predicciones completadas")

print("\n" + "-"*50)
print("MÉTRICAS DE EVALUACIÓN")
print("-"*50)

mse = mean_squared_error(y_train, y_train_pred)
print(f"\n1. Error Cuadrático Medio (MSE):")
print(f"   MSE = {mse:.4f}")
print(f"   Interpretación: Penaliza más los errores grandes")

rmse = np.sqrt(mse)
print(f"\n2. Raíz del Error Cuadrático Medio (RMSE):")
print(f"   RMSE = {rmse:.4f}")
print(f"   Interpretación: Error promedio en la misma escala que la variable objetivo")

mae = mean_absolute_error(y_train, y_train_pred)
print(f"\n3. Error Absoluto Medio (MAE):")
print(f"   MAE = {mae:.4f}")
print(f"   Interpretación: Error promedio absoluto, menos sensible a outliers")

r2 = r2_score(y_train, y_train_pred)
print(f"\n4. Coeficiente de Determinación (R²):")
print(f"   R² = {r2:.4f}")
print(f"   Interpretación: {r2*100:.2f}% de la varianza es explicada por el modelo")

print(f"\n   Calidad del ajuste:")
if r2 >= 0.9:
    print(f"   ✓ Excelente (R² ≥ 0.9)")
elif r2 >= 0.7:
    print(f"   ✓ Bueno (0.7 ≤ R² < 0.9)")
elif r2 >= 0.5:
    print(f"   ⚠ Aceptable (0.5 ≤ R² < 0.7)")
else:
    print(f"   ⚠ Mejorable (R² < 0.5)")

print("\n" + "-"*50)
print("RESUMEN DE MÉTRICAS")
print("-"*50)

metrics_summary = pd.DataFrame({
    'Métrica': ['MSE', 'RMSE', 'MAE', 'R²'],
    'Valor': [mse, rmse, mae, r2]
})

print("\n" + metrics_summary.to_string(index=False))

print("\n" + "-"*50)
print("ANÁLISIS DE ERRORES")
print("-"*50)

errors = y_train - y_train_pred
print(f"\nError promedio: {errors.mean():.4f}")
print(f"Error mínimo: {errors.min():.4f}")
print(f"Error máximo: {errors.max():.4f}")
print(f"Desviación estándar del error: {errors.std():.4f}")

print("\n" + "="*50)
print("EVALUACIÓN COMPLETADA")
print("="*50)

# 5. VALIDACIÓN CRUZADA

print("\n" + "-"*50)
print("5. VALIDACIÓN CRUZADA")
print("-"*50)

print("\nLa validación cruzada evalúa el modelo en diferentes subconjuntos de datos")
print("para obtener una estimación más robusta del rendimiento.")

n_folds = 5
print(f"\nConfiguración: {n_folds}-Fold Cross-Validation")
print(f"  - El dataset se divide en {n_folds} particiones")
print(f"  - El modelo se entrena {n_folds} veces")
print(f"  - En cada iteración, 1 fold se usa para validación y {n_folds-1} para entrenamiento")

print("\n" + "-"*50)
print("EJECUTANDO VALIDACIÓN CRUZADA")
print("-"*50)

print("\nCalculando métricas con validación cruzada...")
print("(Esto puede tomar unos momentos...)")

scoring_metrics = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error',
    'neg_rmse': 'neg_root_mean_squared_error'
}

cv_results = cross_validate(
    model, 
    X_train, 
    y_train, 
    cv=n_folds, 
    scoring=scoring_metrics,
    return_train_score=True,
    n_jobs=-1
)

print("✓ Validación cruzada completada")

cv_r2_scores = cv_results['test_r2']
cv_mse_scores = -cv_results['test_neg_mse']
cv_mae_scores = -cv_results['test_neg_mae']
cv_rmse_scores = -cv_results['test_neg_rmse']

train_r2_scores = cv_results['train_r2']
train_mse_scores = -cv_results['train_neg_mse']

print("\n" + "-"*50)
print("RESULTADOS DE VALIDACIÓN CRUZADA")
print("-"*50)

print(f"\nResultados por fold (validación):")
print(f"\n{'Fold':<6} {'R²':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12}")
print("-" * 52)
for i in range(n_folds):
    print(f"{i+1:<6} {cv_r2_scores[i]:<10.4f} {cv_mse_scores[i]:<12.4f} {cv_rmse_scores[i]:<12.4f} {cv_mae_scores[i]:<12.4f}")

print("\n" + "-"*50)
print("ESTADÍSTICAS DE VALIDACIÓN CRUZADA")
print("-"*50)

print(f"\n1. Coeficiente de Determinación (R²):")
print(f"   Promedio: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
print(f"   Mínimo: {cv_r2_scores.min():.4f}")
print(f"   Máximo: {cv_r2_scores.max():.4f}")

print(f"\n2. Error Cuadrático Medio (MSE):")
print(f"   Promedio: {cv_mse_scores.mean():.4f} ± {cv_mse_scores.std():.4f}")
print(f"   Mínimo: {cv_mse_scores.min():.4f}")
print(f"   Máximo: {cv_mse_scores.max():.4f}")

print(f"\n3. Raíz del Error Cuadrático Medio (RMSE):")
print(f"   Promedio: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
print(f"   Mínimo: {cv_rmse_scores.min():.4f}")
print(f"   Máximo: {cv_rmse_scores.max():.4f}")

print(f"\n4. Error Absoluto Medio (MAE):")
print(f"   Promedio: {cv_mae_scores.mean():.4f} ± {cv_mae_scores.std():.4f}")
print(f"   Mínimo: {cv_mae_scores.min():.4f}")
print(f"   Máximo: {cv_mae_scores.max():.4f}")

print("\n" + "-"*50)
print("ANÁLISIS DE OVERFITTING/UNDERFITTING")
print("-"*50)

train_r2_mean = train_r2_scores.mean()
test_r2_mean = cv_r2_scores.mean()
r2_diff = train_r2_mean - test_r2_mean

print(f"\nR² promedio en entrenamiento: {train_r2_mean:.4f}")
print(f"R² promedio en validación: {test_r2_mean:.4f}")
print(f"Diferencia: {r2_diff:.4f}")

if r2_diff < 0.05:
    print("\n✓ Buen balance - No hay evidencia de overfitting")
elif r2_diff < 0.10:
    print("\n⚠ Ligero overfitting - El modelo generaliza razonablemente bien")
else:
    print("\n⚠ Posible overfitting - El modelo funciona mejor en entrenamiento que en validación")

# Resumen final
print("\n" + "-"*50)
print("RESUMEN DE VALIDACIÓN CRUZADA")
print("-"*50)

cv_summary = pd.DataFrame({
    'Métrica': ['R²', 'MSE', 'RMSE', 'MAE'],
    'Promedio': [
        cv_r2_scores.mean(),
        cv_mse_scores.mean(),
        cv_rmse_scores.mean(),
        cv_mae_scores.mean()
    ],
    'Desv. Estándar': [
        cv_r2_scores.std(),
        cv_mse_scores.std(),
        cv_rmse_scores.std(),
        cv_mae_scores.std()
    ]
})

print("\n" + cv_summary.to_string(index=False))

print("\n" + "="*50)
print("VALIDACIÓN CRUZADA COMPLETADA")
print("="*50)

# 6. COMPARACIÓN DE MÉTRICAS ADICIONALES

print("\n" + "-"*50)
print("6. COMPARACIÓN DE MÉTRICAS ADICIONALES")
print("-"*50)

print("\nEvaluando métricas complementarias para análisis completo del modelo...")

print("\n" + "-"*50)
print("TIEMPO DE ENTRENAMIENTO")
print("-"*50)

print("\nMidiendo tiempo de entrenamiento del modelo...")

model_timing = Ridge(
    alpha=1.0,
    fit_intercept=True,
    max_iter=1000,
    random_state=42
)

start_time = time.time()
model_timing.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"\n✓ Tiempo de entrenamiento: {training_time:.4f} segundos")

if training_time < 1:
    print(f"   Categoría: Muy rápido (< 1 segundo)")
elif training_time < 10:
    print(f"   Categoría: Rápido (1-10 segundos)")
elif training_time < 60:
    print(f"   Categoría: Moderado (10-60 segundos)")
else:
    print(f"   Categoría: Lento (> 60 segundos)")

print("\nMidiendo tiempo de predicción...")
start_time = time.time()
y_pred_timing = model_timing.predict(X_train)
prediction_time = time.time() - start_time

print(f"\n✓ Tiempo de predicción: {prediction_time:.4f} segundos")
print(f"   Predicciones por segundo: {len(X_train)/prediction_time:.0f}")

print("\n" + "-"*50)
print("EFICIENCIA COMPUTACIONAL")
print("-"*50)

print(f"\nCaracterísticas del dataset:")
print(f"  - Muestras de entrenamiento: {X_train.shape[0]:,}")
print(f"  - Número de características: {X_train.shape[1]:,}")
print(f"  - Total de parámetros del modelo: {X_train.shape[1] + 1:,} (coef. + intercepto)")

print(f"\nRelación tiempo/muestra:")
time_per_sample_train = training_time / len(X_train) * 1000
time_per_sample_pred = prediction_time / len(X_train) * 1000
print(f"  - Entrenamiento: {time_per_sample_train:.4f} ms/muestra")
print(f"  - Predicción: {time_per_sample_pred:.4f} ms/muestra")

print(f"\nUso de memoria (estimación):")
memory_coefficients = X_train.shape[1] * 8 / 1024  # 8 bytes por float64
print(f"  - Coeficientes del modelo: ~{memory_coefficients:.2f} KB")

print("\n" + "-"*50)
print("COMPLEJIDAD DEL MODELO")
print("-"*50)

print(f"\nComplejidad temporal:")
print(f"  - Entrenamiento: O(n × p²) donde n = muestras, p = características")
print(f"  - Predicción: O(p) por muestra")

print(f"\nComplejidad espacial:")
print(f"  - Almacenamiento: O(p) para los coeficientes")
print(f"  - Ridge es eficiente en memoria comparado con modelos ensemble")

print("\n" + "-"*50)
print("RESUMEN COMPARATIVO DE MÉTRICAS")
print("-"*50)

final_summary = pd.DataFrame({
    'Categoría': [
        'RENDIMIENTO',
        '',
        '',
        '',
        'VALIDACIÓN CRUZADA',
        '',
        '',
        '',
        'TIEMPO',
        '',
        'COMPLEJIDAD'
    ],
    'Métrica': [
        'R² (Entrenamiento)',
        'MSE',
        'RMSE',
        'MAE',
        'R² (CV Promedio)',
        'MSE (CV Promedio)',
        'RMSE (CV Promedio)',
        'MAE (CV Promedio)',
        'Tiempo Entrenamiento',
        'Tiempo Predicción',
        'Nº Parámetros'
    ],
    'Valor': [
        f"{r2:.4f}",
        f"{mse:.4f}",
        f"{rmse:.4f}",
        f"{mae:.4f}",
        f"{cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}",
        f"{cv_mse_scores.mean():.4f} ± {cv_mse_scores.std():.4f}",
        f"{cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}",
        f"{cv_mae_scores.mean():.4f} ± {cv_mae_scores.std():.4f}",
        f"{training_time:.4f} seg",
        f"{prediction_time:.4f} seg",
        f"{X_train.shape[1] + 1:,}"
    ]
})

print("\n" + final_summary.to_string(index=False))

# Conclusiones finales
print("\n" + "-"*50)
print("CONCLUSIONES")
print("-"*50)

print(f"\n1. Rendimiento predictivo:")
if cv_r2_scores.mean() >= 0.8:
    print(f"   ✓ Excelente capacidad predictiva (R² CV = {cv_r2_scores.mean():.4f})")
elif cv_r2_scores.mean() >= 0.6:
    print(f"   ✓ Buena capacidad predictiva (R² CV = {cv_r2_scores.mean():.4f})")
else:
    print(f"   ⚠ Capacidad predictiva mejorable (R² CV = {cv_r2_scores.mean():.4f})")

print(f"\n2. Generalización:")
if r2_diff < 0.05:
    print(f"   ✓ Excelente generalización (diferencia train-test: {r2_diff:.4f})")
else:
    print(f"   ⚠ Revisar posible overfitting (diferencia train-test: {r2_diff:.4f})")

print(f"\n3. Eficiencia computacional:")
if training_time < 5:
    print(f"   ✓ Entrenamiento muy eficiente ({training_time:.4f} seg)")
else:
    print(f"   ⚠ Considerar optimización si se requiere reentrenamiento frecuente")

print(f"\n4. Escalabilidad:")
print(f"   ✓ Ridge Regression escala bien con el tamaño del dataset")
print(f"   ✓ Bajo uso de memoria ({memory_coefficients:.2f} KB para coeficientes)")
print(f"   ✓ Adecuado para producción con este volumen de datos")

print("\n" + "="*50)
print("ANÁLISIS COMPLETO FINALIZADO")
print("="*50)

# APARTADO 7: AJUSTE DE HIPERPARÁMETROS

print("\n" + "="*50)
print("APARTADO 3: AJUSTE DE HIPERPARÁMETROS")
print("="*50)

print("\n" + "-"*50)
print("1. CONFIGURACIÓN DE LA BÚSQUEDA DE HIPERPARÁMETROS")
print("-"*50)

print("\nEl ajuste de hiperparámetros optimiza el rendimiento del modelo")
print("probando diferentes combinaciones de valores.")

print("\nHiperparámetros de Ridge Regression a ajustar:")
print("  - alpha: Parámetro de regularización L2")
print("  - solver: Algoritmo de optimización")
print("  - max_iter: Número máximo de iteraciones")

# Rendimiento del modelo original (alpha=1.0)
print("\n" + "-"*50)
print("RENDIMIENTO DEL MODELO ORIGINAL")
print("-"*50)

original_metrics = {
    'R² (Train)': r2,
    'MSE (Train)': mse,
    'RMSE (Train)': rmse,
    'MAE (Train)': mae,
    'R² (CV)': cv_r2_scores.mean(),
    'MSE (CV)': cv_mse_scores.mean(),
    'RMSE (CV)': cv_rmse_scores.mean(),
    'MAE (CV)': cv_mae_scores.mean(),
    'Tiempo': training_time
}

print(f"\nModelo original (alpha=1.0):")
print(f"  - R² (Validación Cruzada): {original_metrics['R² (CV)']:.4f}")
print(f"  - RMSE (Validación Cruzada): {original_metrics['RMSE (CV)']:.4f}")
print(f"  - MAE (Validación Cruzada): {original_metrics['MAE (CV)']:.4f}")
print(f"  - Tiempo de entrenamiento: {original_metrics['Tiempo']:.4f} seg")

# Definir grid de hiperparámetros
print("\n" + "-"*50)
print("2. DEFINIENDO GRID DE HIPERPARÁMETROS")
print("-"*50)

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
    'max_iter': [1000, 2000, 5000]
}

print(f"\nGrid de búsqueda definido:")
print(f"  - alpha: {param_grid['alpha']}")
print(f"  - solver: {param_grid['solver']}")
print(f"  - max_iter: {param_grid['max_iter']}")

total_combinations = len(param_grid['alpha']) * len(param_grid['solver']) * len(param_grid['max_iter'])
print(f"\n  Total de combinaciones: {total_combinations}")
print(f"  Con {n_folds}-fold CV: {total_combinations * n_folds} entrenamientos")

# Crear el modelo base
ridge_base = Ridge(random_state=42)

# GridSearchCV
print("\n" + "-"*50)
print("3. EJECUTANDO BÚSQUEDA DE HIPERPARÁMETROS")
print("-"*50)

print("\nIniciando GridSearchCV...")
print("(Esto puede tomar varios minutos...)")

start_time_grid = time.time()

grid_search = GridSearchCV(
    estimator=ridge_base,
    param_grid=param_grid,
    cv=n_folds,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

grid_time = time.time() - start_time_grid

print(f"\n✓ Búsqueda completada en {grid_time:.2f} segundos")

print("\n" + "-"*50)
print("4. MEJORES HIPERPARÁMETROS ENCONTRADOS")
print("-"*50)

print(f"\nMejores hiperparámetros:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\nMejor score (R² en CV): {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

print("\n" + "-"*50)
print("5. EVALUACIÓN DEL MODELO OPTIMIZADO")
print("-"*50)

print("\nEvaluando modelo con hiperparámetros optimizados...")

y_train_pred_opt = best_model.predict(X_train)

mse_opt = mean_squared_error(y_train, y_train_pred_opt)
rmse_opt = np.sqrt(mse_opt)
mae_opt = mean_absolute_error(y_train, y_train_pred_opt)
r2_opt = r2_score(y_train, y_train_pred_opt)

print(f"\nMétricas en conjunto de entrenamiento:")
print(f"  - R²: {r2_opt:.4f}")
print(f"  - MSE: {mse_opt:.4f}")
print(f"  - RMSE: {rmse_opt:.4f}")
print(f"  - MAE: {mae_opt:.4f}")

print("\nRealizando validación cruzada con modelo optimizado...")

cv_results_opt = cross_validate(
    best_model, 
    X_train, 
    y_train, 
    cv=n_folds, 
    scoring=scoring_metrics,
    n_jobs=-1
)

cv_r2_opt = cv_results_opt['test_r2']
cv_mse_opt = -cv_results_opt['test_neg_mse']
cv_rmse_opt = -cv_results_opt['test_neg_rmse']
cv_mae_opt = -cv_results_opt['test_neg_mae']

print(f"\nMétricas de validación cruzada:")
print(f"  - R²: {cv_r2_opt.mean():.4f} ± {cv_r2_opt.std():.4f}")
print(f"  - MSE: {cv_mse_opt.mean():.4f} ± {cv_mse_opt.std():.4f}")
print(f"  - RMSE: {cv_rmse_opt.mean():.4f} ± {cv_rmse_opt.std():.4f}")
print(f"  - MAE: {cv_mae_opt.mean():.4f} ± {cv_mae_opt.std():.4f}")

# Comparación antes y después
print("\n" + "-"*50)
print("6. COMPARACIÓN: MODELO ORIGINAL VS OPTIMIZADO")
print("-"*50)

comparison_df = pd.DataFrame({
    'Métrica': ['R² (Train)', 'MSE (Train)', 'RMSE (Train)', 'MAE (Train)', 
                'R² (CV)', 'MSE (CV)', 'RMSE (CV)', 'MAE (CV)'],
    'Original (α=1.0)': [
        r2, mse, rmse, mae,
        cv_r2_scores.mean(), cv_mse_scores.mean(), 
        cv_rmse_scores.mean(), cv_mae_scores.mean()
    ],
    'Optimizado': [
        r2_opt, mse_opt, rmse_opt, mae_opt,
        cv_r2_opt.mean(), cv_mse_opt.mean(), 
        cv_rmse_opt.mean(), cv_mae_opt.mean()
    ]
})

comparison_df['Mejora (%)'] = (
    (comparison_df['Optimizado'] - comparison_df['Original (α=1.0)']) / 
    comparison_df['Original (α=1.0)'].abs() * 100
)

comparison_df.loc[comparison_df['Métrica'].str.contains('MSE|RMSE|MAE'), 'Mejora (%)'] *= -1

print("\n" + comparison_df.to_string(index=False))

print("\n" + "-"*50)
print("7. ANÁLISIS DE MEJORAS")
print("-"*50)

r2_improvement = ((cv_r2_opt.mean() - cv_r2_scores.mean()) / cv_r2_scores.mean()) * 100
rmse_improvement = ((cv_rmse_scores.mean() - cv_rmse_opt.mean()) / cv_rmse_scores.mean()) * 100
mae_improvement = ((cv_mae_scores.mean() - cv_mae_opt.mean()) / cv_mae_scores.mean()) * 100

print(f"\nMejoras con hiperparámetros optimizados:")
print(f"  1. R² (CV): {r2_improvement:+.2f}%")
if r2_improvement > 0:
    print(f"     ✓ Mejora en capacidad predictiva")
else:
    print(f"     → Sin mejora significativa")

print(f"\n  2. RMSE (CV): {rmse_improvement:+.2f}%")
if rmse_improvement > 0:
    print(f"     ✓ Reducción del error")
else:
    print(f"     → Error similar al original")

print(f"\n  3. MAE (CV): {mae_improvement:+.2f}%")
if mae_improvement > 0:
    print(f"     ✓ Reducción del error absoluto")
else:
    print(f"     → Error similar al original")

print("\n" + "-"*50)
print("8. TOP 5 MEJORES CONFIGURACIONES")
print("-"*50)

cv_results_df = pd.DataFrame(grid_search.cv_results_)
top_5_configs = cv_results_df.nsmallest(5, 'rank_test_score')[
    ['param_alpha', 'param_solver', 'param_max_iter', 'mean_test_score', 'std_test_score']
]

print("\nTop 5 configuraciones por R² en validación cruzada:")
print()
for idx, (_, row) in enumerate(top_5_configs.iterrows(), 1):
    print(f"{idx}. alpha={row['param_alpha']}, solver={row['param_solver']}, "
          f"max_iter={row['param_max_iter']}")
    print(f"   R² = {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
    print()

print("\n" + "-"*50)
print("9. ANÁLISIS DE SENSIBILIDAD - ALPHA")
print("-"*50)

print("\nImpacto del parámetro alpha en el rendimiento:")
alpha_analysis = cv_results_df.groupby('param_alpha')['mean_test_score'].agg(['mean', 'std', 'max'])
alpha_analysis = alpha_analysis.sort_values('mean', ascending=False)

print("\nAlpha\t\tR² Promedio\tDesv. Std\tMejor R²")
print("-" * 60)
for alpha, row in alpha_analysis.iterrows():
    print(f"{alpha}\t\t{row['mean']:.4f}\t\t{row['std']:.4f}\t\t{row['max']:.4f}")

print("\n" + "-"*50)
print("10. CONCLUSIONES DEL AJUSTE DE HIPERPARÁMETROS")
print("-"*50)

print(f"\n1. Mejor configuración encontrada:")
print(f"   - alpha = {grid_search.best_params_['alpha']}")
print(f"   - solver = {grid_search.best_params_['solver']}")
print(f"   - max_iter = {grid_search.best_params_['max_iter']}")

print(f"\n2. Mejora en rendimiento:")
if abs(r2_improvement) < 0.5:
    print(f"   → Mejora marginal ({r2_improvement:+.2f}%)")
    print(f"   → Hiperparámetros originales ya eran adecuados")
elif r2_improvement > 0:
    print(f"   ✓ Mejora significativa de {r2_improvement:.2f}%")
    print(f"   ✓ Recomendado usar modelo optimizado")
else:
    print(f"   → Modelo original es comparable o superior")

print(f"\n3. Costo computacional:")
print(f"   - Tiempo de búsqueda: {grid_time:.2f} segundos")
print(f"   - Combinaciones evaluadas: {total_combinations}")
print(f"   - Tiempo promedio por configuración: {grid_time/total_combinations:.4f} seg")

print(f"\n4. Recomendación final:")
if abs(r2_improvement) >= 1.0:
    print(f"   ✓ Usar modelo optimizado en producción")
    print(f"   ✓ Mejora justifica el costo de búsqueda")
else:
    print(f"   → Modelo original es suficientemente bueno")
    print(f"   → Beneficio marginal del ajuste")

print("\n" + "="*50)
print("AJUSTE DE HIPERPARÁMETROS COMPLETADO")
print("="*50)
