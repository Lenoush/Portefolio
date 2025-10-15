# pages/1_Classification_Poumons.py
import streamlit as st
from PIL import Image
import pandas as pd
from lang import LANG
import numpy as np
import joblib
import cv2

if "lang" not in st.session_state:
    st.session_state.lang = "fr"

col_lang1, col_lang2 = st.columns([0.8, 0.2])
with col_lang2:
    choice = st.radio("🌐", ["FR", "EN"], horizontal=True, label_visibility="collapsed")
    st.session_state.lang = "fr" if choice == "FR" else "en"

t = LANG[st.session_state.lang]

# -------------------------------
# 1. Titre & objectif
# -------------------------------
st.title(t["proj_zoidberg"])
st.write(t["proj_zoidberg_description"])

# -------------------------------
# 2. Exemples de scan
# -------------------------------
st.subheader("Exemples de scans" if st.session_state.lang=="fr" else "Example scans")

cols = st.columns(3)

image_paths = [
    ("Scan Sain", "/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/images/Scan_Sain_Exemple.jpeg"),
    ("Pneumonie Bactérienne", "/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/images/Scan_PneumonieBacterienne.jpeg"),
    ("Pneumonie Virale", "/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/images/Scan_PneumonieViral.jpeg")
]

for col, (caption_fr, path) in zip(cols, image_paths):
    caption = caption_fr if st.session_state.lang == "fr" else caption_fr.replace("Sain","Healthy").replace("Pneumonie Bactérienne","Bacterial Pneumonia").replace("Pneumonie Virale","Viral Pneumonia")
    with col:
        st.image(path, caption=caption, width=200)

# -------------------------------
# 3. Proportion des données / poids
# -------------------------------
st.subheader("Statistiques du dataset" if st.session_state.lang=="fr" else "Dataset statistics")

data_summary = {
    "Classe": ["BACTERIA", "NORMAL", "VIRUS"],
    "Nombre d'images": [2780, 1585, 1493],
    "Poids": [0.702158273381295, 1.2331017056222362, 1.3074346952444742]
}

df = pd.DataFrame(data_summary)
if st.session_state.lang=="en":
    df = df.rename(columns={
        "Classe":"Class",
        "Nombre d'images":"Number of images",
        "Poids":"Weight"
    })

st.dataframe(df)
st.image("/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/images/DatasetPop.png", caption="Distribution des classes" if st.session_state.lang=="fr" else "Class distribution")


# -------------------------------
# 4. Preprocessing
# -------------------------------
st.subheader("Preprocessing des images" if st.session_state.lang=="fr" else "Image preprocessing")

st.write(
    "- Normaliser avec StandardScaler" if st.session_state.lang=="fr"
    else "- Normalize using StandardScaler"
)

st.write(
    "- Redimension des images pour avoir le même cadre et même dimensions." if st.session_state.lang=="fr"
    else "Resizing images to have the same dimensions."
)
# Placeholder pour montrer histogramme dimensions
st.image("/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/images/imageweight.png", caption="Histogramme des dimensions" if st.session_state.lang=="fr" else "2D Histogram of image dimensions")


st.write(
    "- Augmentation des données avec flip horizontal." if st.session_state.lang=="fr"
    else "Data augmentation using horizontal flip."
)
# Placeholder pour montrer flip
st.image("/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/images/flipimage.png", caption="Exemple de flip horizontal" if st.session_state.lang=="fr" else "Example of horizontal flip")


# -------------------------------
# 5. Modèles
# -------------------------------
st.subheader("Modèles testés" if st.session_state.lang=="fr" else "Tested models")

if st.session_state.lang=="fr":
    st.write("""
    ### Approche de modélisation
    
    Plusieurs modèles de classification ont été testés et comparés sur des données prétraitées :
    
    **Optimisation des hyperparamètres :**
    - **GridSearchCV** : Recherche exhaustive sur une grille de paramètres
    - **RandomizedSearchCV** : Recherche aléatoire pour explorer un espace plus large
    
    **Modèles testés :**
    - **SVC (Support Vector Classifier)** avec et sans PCA
    - **Random Forest** avec réduction de dimensionnalité (PCA)
    
    **PCA (Principal Component Analysis)** : Technique de réduction de dimensionnalité qui transforme 
    les features corrélées en composantes principales orthogonales, permettant de réduire la complexité 
    computationnelle tout en conservant la variance maximale des données.
    
    ---

    ### 🏆 Meilleur modèle : SVC avec PCA avec 81% de précision

    Le **Support Vector Classifier avec PCA** a été retenu comme meilleur modèle, entraîné sur les 
    ensembles train et validation avec les paramètres optimaux (γ=0.0001, C=0.1).
    
    **Points forts :**
    - Excellente détection des radiographies **NORMALES**
    - Très bonne identification de la présence de **pneumonie**
    
    **Limites :**
    - Difficulté à différencier précisément **virus** vs **bactérie**

    """)
else:
    st.write("""
    ### Modeling Approach
    
    Several classification models were tested and compared on preprocessed data:
    
    **Hyperparameter optimization:**
    - **GridSearchCV**: Exhaustive grid search over parameter space
    - **RandomizedSearchCV**: Random search to explore a wider parameter space
    
    **Models tested:**
    - **SVC (Support Vector Classifier)** with and without PCA
    - **Random Forest** with dimensionality reduction (PCA)
    
    **PCA (Principal Component Analysis)**: Dimensionality reduction technique that transforms 
    correlated features into orthogonal principal components, reducing computational complexity 
    while preserving maximum data variance.
    
    ---

    ### 🏆 Best Model: SVC with PCA with 81% of accuracy

    The **Support Vector Classifier with PCA** was selected as the best model, trained on train
    and validation sets with optimal parameters (γ=0.0001, C=0.1).
    
    **Strengths:**
    - Excellent detection of **NORMAL** X-rays
    - Very good identification of **pneumonia** presence
    
    **Limitations:**
    - Difficulty in precisely distinguishing **virus** vs **bacteria**
    """)

# Affichage des résultats
col1, col2, col3 = st.columns(3)

with col1:
    st.image("/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/images/svcwithpca.png", 
             caption="Matrice de confusion - Ensemble de validation" if st.session_state.lang=="fr" 
             else "Confusion Matrix - Validation Set")

with col2:
    st.image("/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/images/classification_rapport.png", 
             caption="Rapport de classification" if st.session_state.lang=="fr" 
             else "Classification Report")
    
with col3:
    st.image("/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/images/aucsvcwithpca.png", 
             caption="Courbes ROC" if st.session_state.lang=="fr" 
             else "ROC Curves")

# Métriques clés
st.write("#### Métriques de performance" if st.session_state.lang=="fr" else "#### Performance Metrics")

metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
with metrics_col1:
    st.metric("F1-Score Global" if st.session_state.lang=="fr" else "Overall F1-Score", "80%")
with metrics_col2:
    st.metric("F1-Score NORMAL" if st.session_state.lang=="fr" else "NORMAL F1-Score", "92%", 
              help="Excellent" if st.session_state.lang=="fr" else "Excellent")
with metrics_col3:
    st.metric("F1-Score Pneumonie Bacterienne" if st.session_state.lang=="fr" else "Bacteria Pneumonia F1-Score", "83%",
              help="Bon" if st.session_state.lang=="fr" else "good")
with metrics_col4:
    st.metric("F1-Score Pneumonie Virale" if st.session_state.lang=="fr" else "Viral Pneumonia F1-Score", "62%",
              help="À améliorer" if st.session_state.lang=="fr" else "To improve")
    
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
with metrics_col1:
    st.metric("AUC Global" if st.session_state.lang=="fr" else "Overall AUC", "91.9%")
with metrics_col2:
    st.metric("AUC NORMAL" if st.session_state.lang=="fr" else "NORMAL AUC", "98.6%", 
              help="Excellent" if st.session_state.lang=="fr" else "Excellent")
with metrics_col3:
    st.metric("AUC Pneumonie Bacterienne" if st.session_state.lang=="fr" else "Bacteria Pneumonia AUC", "91.0%",
              help="Bon" if st.session_state.lang=="fr" else "good")
with metrics_col4:
    st.metric("AUC Pneumonie Virale" if st.session_state.lang=="fr" else "Viral Pneumonia AUC", "86.1%",
              help="Bon" if st.session_state.lang=="fr" else "good")
    


# Comparaison des modèles
with st.expander("📊 Comparaison des autres modèles testés" if st.session_state.lang=="fr" 
                 else "📊 Comparison with other tested models"):
    if st.session_state.lang=="fr":
        st.write("""
        | Modèle | PCA | F1-Score | Observations |
        |--------|-----|----------|--------------|
        | **SVC** | ✅ Oui | **79%** | 🏆 Meilleur équilibre |
        | SVC | ❌ Non | ~75% | Bonnes performances mais moins stable |
        | Random Forest | ✅ Oui | ~72% | Plus rapide mais moins précis |
        """)
    else:
        st.write("""
        | Model | PCA | F1-Score | Observations |
        |-------|-----|----------|--------------|
        | **SVC** | ✅ Yes | **79%** | 🏆 Best balance |
        | SVC | ❌ No | ~75% | Good performance but less stable |
        | Random Forest | ✅ Yes | ~72% | Faster but less accurate |
        """)


# -------------------------------
# 6. Interface interactive
# -------------------------------
st.subheader("Tester le modèle" if st.session_state.lang=="fr" else "Test the model")
uploaded = st.file_uploader(
    "Upload un scan de poumon (jpg/png)" if st.session_state.lang=="fr" else "Upload a lung scan (jpg/png)",
    type=["jpg","png"]
)

@st.cache_resource
def load_model():
    return joblib.load("/Users/lenaoudjman/Desktop/Portfolio/projets/zoidberg/models/final_model_svc_pca.pkl")

model = load_model()
classes = ['NORMAL', 'BACTERIA', 'VIRUS'] if st.session_state.lang=="fr" else ['NORMAL', 'BACTERIA', 'VIRUS'] 

IMG_SIZE = 100

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Image importée" if st.session_state.lang=="fr" else "Uploaded image")
    
    img_array = np.array(img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_with_channel = img_normalized.reshape(IMG_SIZE, IMG_SIZE, 1)
    input_vector = img_with_channel.reshape(1, -1)

    # 🧠 Le pipeline applique automatiquement scaler + PCA + SVC
    pred_index = model.predict(input_vector)[0]
    probs = model.predict_proba(input_vector)[0]

    st.success(f"Résultat : **{classes[pred_index]}**")
    prob_dict = {c: float(f"{p:.3f}") for c, p in zip(classes, probs)}
    st.json(prob_dict)
