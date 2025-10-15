# pages/1_Classification_Poumons.py
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

from lang import LANG

t = LANG[st.session_state.lang]

# -------------------------------
st.title(t["proj_zoidberg"])
st.write("""
**Objectif :** prédire si un scan de poumon est sain ou malade.  
Ce projet explore différentes approches : `scikit-learn`, `PyTorch`, et `TensorFlow`.
""")

# --- Exemple visuel des données ---
st.subheader("Exemples de scans")
col1, col2 = st.columns(2)
with col1:
    st.image("data/example_scans/healthy1.jpg", caption="Poumon sain")
with col2:
    st.image("data/example_scans/sick1.jpg", caption="Poumon malade")

# --- Chargement du modèle PyTorch (exemple) ---
@st.cache_resource
def load_model():
    model = torch.load("models/lung_model.pt", map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# --- Interface utilisateur ---
st.subheader("🧠 Teste le modèle")
uploaded = st.file_uploader("Upload un scan de poumon", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Image importée", width=250)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label = "🫁 Sain" if pred == 0 else "⚠️ Malade"
    st.success(f"Résultat : **{label}**")
