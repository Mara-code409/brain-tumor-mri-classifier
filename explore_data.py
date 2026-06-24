import os
from PIL import Image
import matplotlib.pyplot as plt

# Calea catre dataset
data_path = "brisc2025/classification_task/train"

# Afiseaza clasele si numarul de imagini
clase = os.listdir(data_path)
print("Clase disponibile:", clase)

for clasa in clase:
    cale_clasa = os.path.join(data_path, clasa)
    nr_imagini = len(os.listdir(cale_clasa))
    print(f"{clasa}: {nr_imagini} imagini")

# Afiseaza cate o imagine din fiecare clasa
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, clasa in enumerate(clase):
    cale_clasa = os.path.join(data_path, clasa)
    prima_imagine = os.listdir(cale_clasa)[0]
    img = Image.open(os.path.join(cale_clasa, prima_imagine))
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(clasa)
    axes[i].axis('off')

plt.suptitle("BRISC 2025 - Exemple din fiecare clasa")
plt.tight_layout()
plt.show()