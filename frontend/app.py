import streamlit as st
import sys
import os
from PIL import Image
import tempfile
import time

# Import de ton backend
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ocr.predict import PredictionConfig, load_trained_model, predict_ocr


# ============================================================================
# CONFIGURATION PAGE
# ============================================================================
st.set_page_config(
    page_title="OCR Handwritten Algorithm",
    page_icon="🔤",
    layout="centered"
)

# ============================================================================
# CACHE DU MODÈLE (IMPORTANT POUR PERFORMANCE)
# ============================================================================

@st.cache_resource
def load_model_once():
    config = PredictionConfig()
    model, char_to_num, num_to_char = load_trained_model(config)
    return model, char_to_num, num_to_char

model, char_to_num, num_to_char = load_model_once()

# ============================================================================
# INTERFACE
# ============================================================================

st.title("🔤 Handwritten Algorithm → Python")
st.write("Transformez vos algorithmes manuscrits en code Python")

uploaded_file = st.file_uploader(
    "📸 Upload une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Afficher image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée", width="stretch")

    # Créer UN SEUL fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        uploaded_file_path = tmp_file.name

    if st.button("🔮 Reconnaître l'algorithme"):

        with st.spinner("Analyse en cours..."):

            try:
                start = time.time()

                result = predict_ocr(
                    model=model,
                    num_to_char=num_to_char,
                    image_path=uploaded_file_path
                )

                duration = round(time.time() - start, 2)

                st.success(f"Reconnaissance réussie en {duration}s")
                st.code(result, language="python")

                st.download_button(
                    label="📥 Télécharger le code Python",
                    data=result,
                    file_name="generated_code.py",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Erreur : {str(e)}")