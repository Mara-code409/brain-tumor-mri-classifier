import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np

# Incarca modelul
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load('model_efficientnet.pth', map_location='cpu'))
model.eval()

clase = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.title("Brain Tumor MRI Classification")
st.write("Incarca o imagine MRI pentru a detecta tipul de tumora.")

uploaded_file = st.file_uploader("Alege o imagine MRI", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Imaginea incarcata", width=300)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predictie = output.argmax().item()

    st.subheader(f"Predictie: {clase[predictie]}")
    st.write("Probabilitati:")
    for i, clasa in enumerate(clase):
        st.progress(float(probabilities[i]), text=f"{clasa}: {probabilities[i]*100:.1f}%")

    # GradCAM
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor)[0]

    img_array = np.array(img.resize((224, 224))) / 255.0
    vizualizare = show_cam_on_image(img_array.astype(np.float32), grayscale_cam)

    st.subheader("GradCAM - Unde s-a uitat modelul")
    st.image(vizualizare, width=300)