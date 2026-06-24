import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Incarca modelul pre-trained salvat
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load('model_efficientnet.pth'))
model.eval()

clase = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Transformari
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Incarca o imagine de test
dataset = ImageFolder(root='brisc2025/classification_task/test', transform=transform)
imagine_tensor, eticheta_reala = dataset[0]
imagine_tensor = imagine_tensor.unsqueeze(0)

# GradCAM
target_layer = model.features[-1]
cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=imagine_tensor)[0]

# Imaginea originala pentru vizualizare
imagine_originala = dataset.imgs[0][0]
img = Image.open(imagine_originala).resize((224, 224))
img_array = np.array(img) / 255.0
if len(img_array.shape) == 2:
    img_array = np.stack([img_array]*3, axis=-1)

# Suprapune GradCAM pe imagine
vizualizare = show_cam_on_image(img_array.astype(np.float32), grayscale_cam)

# Predictie
with torch.no_grad():
    output = model(imagine_tensor)
    predictie = output.argmax().item()

# Afiseaza
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title(f"Original\nClasa reala: {clase[eticheta_reala]}")
axes[0].axis('off')

axes[1].imshow(vizualizare)
axes[1].set_title(f"GradCAM\nPredictie: {clase[predictie]}")
axes[1].axis('off')

plt.tight_layout()
plt.show()