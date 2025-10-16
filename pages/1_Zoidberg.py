import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import cv2
import os
from langues.lang_zoidberg import LANG_ZOIDBERG

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(layout="wide")

# ------------------------------------------------------
# Langue
# ------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "fr"

col_lang1, col_lang2 = st.columns([0.8, 0.2])
with col_lang2:
    choice = st.radio("🌐", ["FR", "EN"], horizontal=True, label_visibility="collapsed")
    st.session_state.lang = "fr" if choice == "FR" else "en"

t = LANG_ZOIDBERG[st.session_state.lang]

# ------------------------------------------------------
# Titre et description
# ------------------------------------------------------
st.title(t["proj_zoidberg"])
st.write(t["proj_zoidberg_description"])

# ------------------------------------------------------
# Métriques clés
# ------------------------------------------------------
st.subheader(t["metrics_title"])

for row in t["metrics_rows"]:
    cols = st.columns(4)
    for col, metric in zip(cols, row):
        with col:
            st.metric(metric["label"], metric["value"], help=metric.get("help", None))

# ------------------------------------------------------
# Interface de test
# ------------------------------------------------------
st.subheader(t["test_model_title"])

# Images prédéfinies
healthy_dir = Path(os.path.join(BASE_DIR, "../projets/zoidberg/images/healthy"))
bacterial_dir = Path(os.path.join(BASE_DIR, "../projets/zoidberg/images/bacterial"))
viral_dir = Path(os.path.join(BASE_DIR, "../projets/zoidberg/images/viral"))

healthy_images = [None] + sorted(healthy_dir.glob("*.jpeg"))
bacterial_images = [None] + sorted(bacterial_dir.glob("*.jpeg"))
viral_images = [None] + sorted(viral_dir.glob("*.jpeg"))

col1, col2, col3, col4 = st.columns(4)

# Fonction de réinitialisation
def reset_selection():
    st.session_state["healthy_select"] = None
    st.session_state["bacterial_select"] = None
    st.session_state["viral_select"] = None

with col1:
    selected_healthy = st.selectbox(
        t["healthy_label"], healthy_images,
        format_func=lambda x: x.name if x else "--",
        key="healthy_select"
    )
with col2:
    selected_bacterial = st.selectbox(
        t["bacterial_label"], bacterial_images,
        format_func=lambda x: x.name if x else "--",
        key="bacterial_select"
    )
with col3:
    selected_viral = st.selectbox(
        t["viral_label"], viral_images,
        format_func=lambda x: x.name if x else "--",
        key="viral_select"
    )
with col4:
    st.button(t["reset_button"], on_click=reset_selection)

uploaded = st.file_uploader(t["upload_label"], type=["jpg", "png", "jpeg"])

# ------------------------------------------------------
# Chargement du modèle
# ------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "../projets/zoidberg/models/final_model_svc_pca.pkl"))

model = load_model()
classes = ["NORMAL", "BACTERIA", "VIRUS"]

# ------------------------------------------------------
# Prédiction
# ------------------------------------------------------
IMG_SIZE = 100

def predict_image(image_path_or_obj):
    img = Image.open(image_path_or_obj).convert("RGB")
    img_array = np.array(img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_vector = img_normalized.reshape(1, -1)

    probs = model.predict_proba(input_vector)[0]
    return probs, img

# Sélection finale
image_to_test = None
expected_class = None

if uploaded:
    image_to_test = uploaded
    expected_class = None
    st.info(t["info_uploaded"])
elif selected_healthy:
    image_to_test = selected_healthy
    expected_class = "NORMAL"
elif selected_bacterial:
    image_to_test = selected_bacterial
    expected_class = "BACTERIA"
elif selected_viral:
    image_to_test = selected_viral
    expected_class = "VIRUS"

# Affichage résultat
if image_to_test:
    probs, img = predict_image(image_to_test)
    st.image(img, caption=t["uploaded_image_caption"], width=200)
    prob_dict = {c: float(f"{p:.3f}") for c, p in zip(classes, probs)}
    st.write(t["Show_Proba"])
    st.json(prob_dict)

    if expected_class:
        predicted_class = classes[np.argmax(probs)]
        if predicted_class == expected_class:
            st.success(t["correct_result"].format(expected_class=expected_class))
        else:
            st.error(t["wrong_result"].format(predicted_class=predicted_class, expected_class=expected_class))

# ------------------------------------------------------
# Données & préprocessing
# ------------------------------------------------------
st.subheader(t["preprocess_data"])
st.write(t["example_scans_title"])

cols = st.columns(3)
for col, img in zip(cols, t["example_scans"]):
    with col:
        st.image(os.path.join(BASE_DIR, img["path"]), caption=img["caption"], width=200)

st.write(t["dataset_title"])
df = pd.DataFrame(t["dataset"])
st.dataframe(df)
st.image(os.path.join(BASE_DIR, "../projets/zoidberg/images/DatasetPop.png"), caption=t["dataset_image_caption"])

st.write(t["preprocessing_title"])
for step in t["preprocessing_steps"]:
    st.write(step["text"])
    if step["image"]:
        cols = st.columns(len(step["image"]))
        for col, img_path in zip(cols, step["image"]):
            with col:
                st.image(os.path.join(BASE_DIR, img_path), caption=step.get("caption", ""))

# ------------------------------------------------------
# Modélisation
# ------------------------------------------------------
st.subheader(t["models_title"])
st.write(t["modeling_text"])

cols = st.columns(3)
for col, img in zip(cols, t["model_images"]):
    with col:
        st.image(os.path.join(BASE_DIR, img["path"]), caption=img["caption"])

with st.expander(t["comparison_title"]):
    st.markdown(t["comparison_table"])

st.caption(t["footer"])
