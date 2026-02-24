import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

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
    optimizer="adam",                          
    loss="sparse_categorical_crossentropy",    
    metrics=["accuracy"]                       
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

# ── TensorBoard callback ─────────────────────────────────────────────────────
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,       # histogramas de pesos cada época
    write_graph=True,
    write_images=False,
)
print(f"\n✓ TensorBoard logs → {log_dir}")
print(f"  Lanza TensorBoard con: tensorboard --logdir logs")

print("\nEntrenando el modelo…")
historial = modelo.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
    callbacks=[tensorboard_cb]
)

loss_test, acc_test = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\nResultados en test → Pérdida: {loss_test:.4f} | Accuracy: {acc_test*100:.2f}%")

# ── Curvas de aprendizaje ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(historial.history["accuracy"],     label="Entrenamiento")
axes[0].plot(historial.history["val_accuracy"], label="Validación")
axes[0].set_title("Accuracy por época")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(historial.history["loss"],     label="Entrenamiento")
axes[1].plot(historial.history["val_loss"], label="Validación")
axes[1].set_title("Pérdida por época")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Curvas de Aprendizaje – CNN MNIST", fontsize=14)
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()
print("✓ Curvas de aprendizaje guardadas en learning_curves.png")

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
print("\nInforme de clasificación por dígito:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
