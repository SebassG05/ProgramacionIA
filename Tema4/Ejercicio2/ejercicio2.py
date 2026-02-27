import pandas as pd
import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, cross_validate

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
