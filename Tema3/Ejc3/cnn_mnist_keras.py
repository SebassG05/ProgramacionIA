import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

modelo = Sequential([

    Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
           padding="same", input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(units=256, activation="relu"),
    Dropout(rate=0.5),          
    Dense(units=128, activation="relu"),
    Dense(units=10, activation="softmax"),  
])

print("Arquitectura de la CNN:")
modelo.summary()
print("\n✓ Arquitectura definida correctamente con la API Secuencial de Keras.")

# ── 2. Función de pérdida, optimizador y métricas ─────────────────────────────
modelo.compile(
    optimizer="adam",                          # Optimizador Adam (lr=0.001 por defecto)
    loss="sparse_categorical_crossentropy",    # Pérdida para etiquetas enteras
    metrics=["accuracy"]                       # Métrica de evaluación: accuracy
)
print("\n✓ Modelo compilado con:")
print("   · Optimizador : Adam")
print("   · Pérdida     : sparse_categorical_crossentropy")
print("   · Métrica     : accuracy")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar [0,255] → [0,1] y añadir canal (28,28) → (28,28,1)
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0
X_train = X_train[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

print("\nEntrenando el modelo…")
historial = modelo.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

loss_test, acc_test = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\nResultados en test → Pérdida: {loss_test:.4f} | Accuracy: {acc_test*100:.2f}%")

y_pred = np.argmax(modelo.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=list(range(10)))
fig, ax = plt.subplots(figsize=(9, 9))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Matriz de Confusión – CNN MNIST", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("\n✓ Matriz de confusión guardada en confusion_matrix.png")
