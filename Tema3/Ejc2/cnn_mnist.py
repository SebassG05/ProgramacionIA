import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()

        self.bloque_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   
        )

        self.bloque_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   
        )

        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.bloque_conv1(x)
        x = self.bloque_conv2(x)
        x = self.clasificador(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

modelo = CNN_MNIST().to(device)
print("\nArquitectura de la CNN:")
print(modelo)

batch_prueba = torch.zeros(8, 1, 28, 28).to(device)   
salida_prueba = modelo(batch_prueba)
print(f"\nEntrada: {batch_prueba.shape}")
print(f"Salida:  {salida_prueba.shape}  (esperado: [8, 10])")
print("\n✓ Arquitectura definida correctamente.")

# Función de pérdida y optimizador

criterio    = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

print("\nFunción de pérdida: CrossEntropyLoss")
print("Optimizador:        Adam (lr=0.001)")
print("\n✓ Optimizador y función de pérdida definidos correctamente.")

# Carga del dataset MNIST

transformacion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transformacion)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transformacion)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f"Imágenes de entrenamiento: {len(train_dataset)}")
print(f"Imágenes de prueba:        {len(test_dataset)}")

# Bucle de entrenamiento

EPOCHS = 5
print(f"\nEntrenando durante {EPOCHS} épocas...\n")

for epoca in range(1, EPOCHS + 1):
    modelo.train()
    perdida_total = 0.0

    for imagenes, etiquetas in train_loader:
        imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)

        optimizador.zero_grad()                       
        salidas = modelo(imagenes)                    
        perdida = criterio(salidas, etiquetas)        
        perdida.backward()                             
        optimizador.step()                             

        perdida_total += perdida.item()

    perdida_media = perdida_total / len(train_loader)
    print(f"  Época {epoca}/{EPOCHS}  |  Pérdida media: {perdida_media:.4f}")

print("\n✓ Entrenamiento completado.")

# Visualización de características aprendidas

modelo.eval()

filtros = modelo.bloque_conv1[0].weight.data.cpu()  # shape: (32, 1, 3, 3)

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
fig.suptitle("Filtros aprendidos — Conv1 (32 filtros de 3×3)", fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.imshow(filtros[i, 0], cmap="viridis")
    ax.axis("off")
plt.tight_layout()
plt.savefig("filtros_conv1.png", dpi=150)
plt.show()
print("✓ Filtros de Conv1 guardados en filtros_conv1.png")

imagen_muestra, etiqueta_muestra = test_dataset[0]
imagen_muestra = imagen_muestra.unsqueeze(0).to(device)  # (1, 1, 28, 28)

with torch.no_grad():
    act_conv1 = modelo.bloque_conv1(imagen_muestra)           # (1, 32, 14, 14)
    act_conv2 = modelo.bloque_conv2(act_conv1)                # (1, 64, 7, 7)

act1 = act_conv1.squeeze(0).cpu()  # (32, 14, 14)
fig, axes = plt.subplots(4, 8, figsize=(14, 7))
fig.suptitle(f"Activaciones Conv1 — dígito '{etiqueta_muestra}'", fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.imshow(act1[i], cmap="inferno")
    ax.axis("off")
plt.tight_layout()
plt.savefig("activaciones_conv1.png", dpi=150)
plt.show()
print("✓ Activaciones de Conv1 guardadas en activaciones_conv1.png")

act2 = act_conv2.squeeze(0).cpu()  # (64, 7, 7)
fig, axes = plt.subplots(8, 8, figsize=(14, 14))
fig.suptitle(f"Activaciones Conv2 — dígito '{etiqueta_muestra}'", fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.imshow(act2[i], cmap="inferno")
    ax.axis("off")
plt.tight_layout()
plt.savefig("activaciones_conv2.png", dpi=150)
plt.show()
print("✓ Activaciones de Conv2 guardadas en activaciones_conv2.png")

# Evaluación del modelo: accuracy y matriz de confusión

modelo.eval()
todas_predicciones = []
todas_etiquetas    = []

with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        imagenes = imagenes.to(device)
        salidas  = modelo(imagenes)
        _, predicciones = torch.max(salidas, 1)
        todas_predicciones.extend(predicciones.cpu().numpy())
        todas_etiquetas.extend(etiquetas.numpy())

accuracy = np.mean(np.array(todas_predicciones) == np.array(todas_etiquetas))
print(f"\nAccuracy en test: {accuracy*100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(todas_etiquetas, todas_predicciones, digits=4))

cm = confusion_matrix(todas_etiquetas, todas_predicciones)
print("Matriz de confusión:")
print(cm)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Matriz de confusión — CNN MNIST")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("✓ Matriz de confusión guardada en confusion_matrix.png")

