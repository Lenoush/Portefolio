import streamlit as st
from lang import LANG
from pathlib import Path

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

col1, col2, col3 = st.columns([1, 1, 1], gap="large")
with col1:
    with open(cv_path, "rb") as f:
        cv_bytes = f.read()

    st.download_button(
        label=t["cv"],
        data=cv_bytes,
        file_name="CV_Léna_Oudjman_2025.pdf",
        mime="application/pdf"
    )
with col2:
    st.link_button(t["linkedin"], "https://www.linkedin.com/in/lena-oudjman-0a36b6226/")
with col3:
    st.link_button(t["github"], "https://github.com/Lenoush")

st.markdown(f"""
{t['location']}  
{t['email']}  
{t['phone']}
""")

st.divider()

# -------------------------------
# MENU DES PROJETS
# -------------------------------

# Cards avec descriptions (plus visuelles)
st.markdown("### 💼 Explorer mes projets")

project1_col, project2_col = st.columns(2)


with project1_col:
    with st.container(border=True):
        st.markdown(f"#### 🫁 {t['proj_zoidberg']}")
        st.write("Deep Learning pour la classification d'images médicales")
        if st.button("Voir le projet →", key="proj1", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Zoidberg.py")
            st.rerun()  

with project2_col:
    with st.container(border=True):
        st.markdown("#### 🔬 Autre Projet")
        st.write("Description de votre autre projet")
        if st.button("Voir le projet →", key="proj2", use_container_width=True):
            st.switch_page("pages/2_Autre.py")

st.divider()
st.caption(t["footer"])
