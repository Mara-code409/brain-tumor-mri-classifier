# Brain Tumor Classification from MRI Images

Four-class brain tumor classifier built on a fine-tuned **EfficientNet-B0** convolutional neural network. The model sorts brain MRI scans into **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**, and reaches **98.10% test accuracy** (macro F1 = 0.98) when fine-tuned from ImageNet weights.

This repository accompanies a short IEEE-style paper describing the project.

## Results

| Training regime | Test accuracy |
|---|---|
| EfficientNet-B0, pre-trained (ImageNet) | **98.10%** |
| EfficientNet-B0, trained from scratch | 79.60% |

The ~18-point gap shows how much transfer learning matters on a small medical dataset. Grad-CAM was used to confirm the model attends to the tumor region, and a Streamlit app provides interactive inference.

## Dataset

The project uses the **BRISC 2025** dataset (6000 T1-weighted MRI images, 5000 train / 1000 test, balanced across the four classes). It is **not included** in this repository because of its size.

Download it from Kaggle and place it in the project root so the folder structure looks like this:

```
brisc2025/classification_task/train/{glioma,meningioma,no_tumor,pituitary}/
brisc2025/classification_task/test/{glioma,meningioma,no_tumor,pituitary}/
```

## Project structure

| File | What it does |
|---|---|
| `explore_data.py` | Prints class counts and shows one sample image per class |
| `train.py` | Trains EfficientNet-B0 from scratch and saves `model_efficientnet_notpretrained.pth` |
| `metrici.py` | Trains the pre-trained model, prints the classification report, and plots loss/accuracy curves + confusion matrix |
| `gradcam.py` | Runs Grad-CAM on a test image to visualize where the model looks |
| `app.py` | Streamlit web app for interactive inference with Grad-CAM overlay |
| `model_efficientnet.pth` | Trained pre-trained model weights (used by `gradcam.py` and `app.py`) |
| `model_efficientnet_notpretrained.pth` | Weights of the from-scratch model |

## Setup

```bash
# (optional) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## How to run

Run all commands from the project root (the folder that contains `brisc2025/`).

```bash
python explore_data.py        # inspect the dataset
python metrici.py             # train pre-trained model + metrics and plots
python train.py               # train the from-scratch baseline
python gradcam.py             # Grad-CAM visualization on a test image
streamlit run app.py          # launch the interactive web app
```

## Model

- **Backbone:** EfficientNet-B0 (`torchvision.models`)
- **Head:** final layer replaced with a 4-class linear layer
- **Optimizer:** Adam, learning rate 0.001
- **Loss:** cross-entropy
- **Batch size:** 32, **epochs:** 5
- **Preprocessing:** resize to 224×224, ToTensor, normalize with ImageNet mean/std

## Author

Tănase Mara-Ruxandra — Faculty of Mathematics and Computer Science, University of Bucharest.
