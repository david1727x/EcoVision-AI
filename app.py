import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="EcoVision AI", page_icon="♻️", layout="centered")


class EcoCNN(nn.Module):
    def __init__(self, n_clases):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 96), nn.ReLU(), nn.Dropout(0.35), nn.Linear(96, n_clases)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def construir_modelo(nombre, n_clases):
    if nombre == "cnn":
        return EcoCNN(n_clases)
    if nombre == "resnet18":
        modelo = models.resnet18(weights=None)
        modelo.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(modelo.fc.in_features, n_clases))
        return modelo
    raise ValueError(f"Arquitectura desconocida: {nombre}")


@st.cache_resource
def cargar_modelo(ruta):
    pack = torch.load(ruta, map_location="cpu", weights_only=True)
    mod = construir_modelo(pack["architecture"], len(pack["classes"]))
    mod.load_state_dict(pack["model_state_dict"])
    mod.eval()
    procesar = transforms.Compose([
        transforms.Resize((pack["image_size"], pack["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(pack["mean"], pack["std"])
    ])
    return mod, pack, procesar


st.title("♻️ EcoVision AI")
st.write("Clasificación exploratoria de **residuos** mediante redes neuronales convolucionales.")

# Acepta tanto la ruta raíz (subida desde GitHub Web) como artifacts/.
CHECKPOINT = Path(__file__).parent / "artifacts" / "best_model.pth"
if not CHECKPOINT.exists():
    CHECKPOINT = Path(__file__).parent / "best_model.pth"
if not CHECKPOINT.exists():
    st.warning("Falta el modelo entrenado: sube best_model.pth a la raíz del repositorio o a artifacts/.")
    st.stop()

try:
    modelo, config, procesar = cargar_modelo(str(CHECKPOINT))
except Exception as exc:
    st.error(f"No se pudo cargar el modelo: {exc}")
    st.stop()

traduccion = {
    "cardboard": "Cartón", "glass": "Vidrio", "metal": "Metal",
    "paper": "Papel", "plastic": "Plástico", "trash": "Basura general"
}
st.caption(
    f'Modelo: {config["architecture"]} · {len(config["classes"])} categorías · '
    f'{"modo rápido" if config.get("modo_rapido") else "entrenamiento completo"}'
)
archivo = st.file_uploader("Sube una fotografía de un solo residuo", type=["jpg", "jpeg", "png", "webp"])

if archivo:
    try:
        imagen = Image.open(io.BytesIO(archivo.read())).convert("RGB")
        st.image(imagen, caption="Imagen cargada", use_container_width=True)
        if st.button("Clasificar imagen", type="primary"):
            x = procesar(imagen).unsqueeze(0)
            with torch.inference_mode():
                prob = torch.softmax(modelo(x), dim=1).squeeze(0).cpu().numpy()
            orden = np.argsort(prob)[::-1]
            pred = int(orden[0])
            st.success(
                f'Resultado: **{traduccion.get(config["classes"][pred], config["classes"][pred])}**'
            )
            st.metric("Confianza del modelo (no calibrada)", f"{prob[pred]:.1%}")
            tabla = pd.DataFrame({
                "Categoría": [traduccion.get(config["classes"][i], config["classes"][i]) for i in orden],
                "Probabilidad": [float(prob[i]) for i in orden]
            }).set_index("Categoría")
            st.bar_chart(tabla)
    except (OSError, ValueError) as exc:
        st.error(f"Archivo de imagen inválido: {exc}")

st.info(
    "Prototipo académico. La confianza mostrada no equivale a una probabilidad calibrada. "
    "Las predicciones pueden fallar con fotos fuera del dataset, objetos múltiples o materiales combinados."
)
