import streamlit as st
from langues.lang import LANG
from pathlib import Path

st.set_page_config(layout="wide")

# -------------------------------
# CHOIX DE LA LANGUE
# -------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "fr"

col_lang1, col_lang2 = st.columns([0.8, 0.2])
with col_lang2:
    choice = st.radio("🌐", ["FR", "EN"], horizontal=True, label_visibility="collapsed")
    st.session_state.lang = "fr" if choice == "FR" else "en"

t = LANG[st.session_state.lang]

# -------------------------------
# CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Portfolio - Léna Oudjman",
    page_icon="💼",
    layout="centered"
)

# -------------------------------
# CONTENU
# -------------------------------
st.title(t["title"])
st.markdown(f"### {t['subtitle']}")
st.divider()

cv_path = Path("images/CV_Léna_Oudjman_2025.pdf")

with open(cv_path, "rb") as f:
    cv_bytes = f.read()

# 2 colonnes
col_left, col_right = st.columns([1, 1], gap="large")

# --- Ligne 1 ---
with col_left:
    st.write(t["location"])
with col_right:
    st.link_button(t["linkedin"], "https://www.linkedin.com/in/lena-oudjman-0a36b6226/")

# --- Ligne 2 ---
with col_left:
    st.write(t["email"])
with col_right:
    st.link_button(t["github"], "https://github.com/Lenoush")

# --- Ligne 3 ---
with col_left:
    st.write(t["phone"])
with col_right:
    st.download_button(
        label=t["cv"],
        data=cv_bytes,
        file_name="CV_Léna_Oudjman_2025.pdf",
        mime="application/pdf"
    )

st.divider()

# -------------------------------
# MENU DES PROJETS
# -------------------------------

# Cards avec descriptions (plus visuelles)
st.markdown(f"### 💼 {t['click_project']}")


project1_col, project2_col = st.columns(2)


with project1_col:
    with st.container(border=True):
        st.markdown(f"#### {t['proj_zoidberg']}")
        st.write(t["proj_zoidberg_description"])
        if st.button(t["place_to_click_on"], key="proj1", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Zoidberg.py")
            st.rerun()  

with project2_col:
    with st.container(border=True):
        st.markdown("#### 🔬 Autre Projet")
        st.write("Description de votre autre projet")
        if st.button(t["place_to_click_on"], key="proj2", use_container_width=True):
            st.switch_page("pages/2_Autre.py")

st.divider()
st.caption(t["footer"])
