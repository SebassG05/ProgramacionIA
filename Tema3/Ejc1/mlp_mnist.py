from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

#Carga del dataset MNIST

print("Cargando dataset MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

# Separamos características (píxeles) y etiquetas (dígitos 0-9)
X = mnist.data    
y = mnist.target  

print(f"  Forma de X (imágenes): {X.shape}")
print(f"  Forma de y (etiquetas): {y.shape}")
print(f"  Rango de valores ANTES de normalizar: [{X.min()}, {X.max()}]")

# Normalización de píxeles

X = X / 255.0

print(f"  Rango de valores DESPUÉS de normalizar: [{X.min():.2f}, {X.max():.2f}]")
print("\n✓ Datos cargados y normalizados correctamente.")

# División en entrenamiento (80%) y prueba (20%)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n  Entrenamiento: {X_train.shape[0]} imágenes")
print(f"  Prueba:        {X_test.shape[0]} imágenes")
print("\n✓ División completada correctamente.")

# Entrenamiento con diferentes configuraciones de hiperparámetros

configuraciones = [
    {
        "nombre": "Config 1 — 1 capa (100), relu, adam",
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 20,
    },
    {
        "nombre": "Config 2 — 2 capas (256, 128), relu, adam",
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 20,
    },
    {
        "nombre": "Config 3 — 2 capas (128, 64), tanh, sgd",
        "hidden_layer_sizes": (128, 64),
        "activation": "tanh",
        "solver": "sgd",
        "max_iter": 20,
    },
    {
        "nombre": "Config 4 — 3 capas (512, 256, 128), relu, adam",
        "hidden_layer_sizes": (512, 256, 128),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 20,
    },
]

resultados = []

for cfg in configuraciones:
    print(f"\nEntrenando: {cfg['nombre']} ...")
    modelo = MLPClassifier(
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        activation=cfg["activation"],
        solver=cfg["solver"],
        max_iter=cfg["max_iter"],
        random_state=42,
        verbose=False,
    )
    modelo.fit(X_train, y_train)
    accuracy = modelo.score(X_test, y_test)
    resultados.append({"nombre": cfg["nombre"], "accuracy": accuracy, "modelo": modelo})
    print(f"  → Accuracy en test: {accuracy:.4f} ({accuracy*100:.2f}%)")

mejor = max(resultados, key=lambda r: r["accuracy"])
print(f"\n✓ Mejor configuración: {mejor['nombre']}")
print(f"  Accuracy: {mejor['accuracy']*100:.2f}%")

# Evaluación del mejor modelo

print("\n--- Evaluación detallada del mejor modelo ---")
y_pred = mejor["modelo"].predict(X_test)

print(f"\nAccuracy: {mejor['accuracy']*100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Matriz de confusión — {mejor['nombre']}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("\n✓ Matriz de confusión guardada en confusion_matrix.png")
