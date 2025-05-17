import streamlit as st
import cv2
import numpy as np
import time
import os
from geopy.distance import geodesic
from streamlit_javascript import st_javascript
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# --- Configuration initiale ---
st.set_page_config(page_title="Contrôle de Présence", layout="centered")
st.title("📍 Application de Contrôle de Présence")

# Dossier pour stocker les images
if not os.path.exists("photos"):
    os.makedirs("photos")

log_path = "journal_presence.csv"

# Initialiser la position de base
if "base_location" not in st.session_state:
    st.session_state.base_location = None

# --- Fonctions ---
def get_real_location():
    coords = st_javascript("""
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const lat = pos.coords.latitude;
                const lon = pos.coords.longitude;
                Streamlit.setComponentValue([lat, lon]);
            },
            (err) => {
                Streamlit.setComponentValue(null);
            }
        );
    """)
    return tuple(coords) if coords else None

def capture_image():
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def enregistrer_presence(tel, location, distance, status):
    log_data = {
        "Telephone": tel,
        "DateHeure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Latitude": location[0],
        "Longitude": location[1],
        "Distance_m": int(distance),
        "Statut": status
    }
    df = pd.DataFrame([log_data])
    if not os.path.exists(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode='a', header=False, index=False)

def verifier_visage(ref_path, new_img):
    try:
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
        result = DeepFace.verify(img1_path=ref_path, img2_path=temp_path, enforce_detection=True)
        os.remove(temp_path)
        return result["verified"]
    except Exception as e:
        st.error(f"Erreur DeepFace : {e}")
        return False

# --- Interface ---
menu = ["📷 Enregistrement", "✅ Vérification", "📊 Journal de Présence"]
choice = st.sidebar.radio("Navigation", menu)

# Enregistrement de l'agent
if choice == "📷 Enregistrement":
    st.subheader("Étape 1 : Enregistrement de l'agent")
    tel = st.text_input("Numéro de téléphone de l'agent", max_chars=8)

    if len(tel) == 8 and tel.isdigit():
        location = get_real_location()
        if location:
            st.success(f"📌 Coordonnées détectées automatiquement : {location}")
            st.session_state.base_location = location
        else:
            st.warning("⚠️ GPS indisponible. Entrez manuellement :")
            lat = st.number_input("Latitude manuelle", value=6.1319, format="%.6f")
            lon = st.number_input("Longitude manuelle", value=1.2228, format="%.6f")
            location = (lat, lon)
            st.session_state.base_location = location

        if st.button("📸 Capturer et Enregistrer"):
            img = capture_image()
            if img is not None:
                cv2.imwrite(f"photos/{tel}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                st.image(img, caption="Image capturée", use_container_width=True)
                st.success("✅ Agent enregistré avec succès.")
                st.info(f"Coordonnées de référence : {location}")
            else:
                st.error("Erreur de capture.")
    else:
        st.warning("Numéro invalide (8 chiffres).")

# Vérification
elif choice == "✅ Vérification":
    st.subheader("Étape 2 : Vérification de la présence")
    tel = st.text_input("Numéro de téléphone de l'agent", max_chars=8)

    if len(tel) == 8 and tel.isdigit():
        path = f"photos/{tel}.jpg"
        if not os.path.exists(path):
            st.error("❌ Aucun enregistrement pour ce numéro.")
        else:
            location = get_real_location()
            if location:
                st.success(f"📌 Position actuelle : {location}")
            else:
                st.warning("⚠️ GPS indisponible. Entrez manuellement :")
                lat = st.number_input("Latitude actuelle", value=6.1319, format="%.6f")
                lon = st.number_input("Longitude actuelle", value=1.2228, format="%.6f")
                location = (lat, lon)

            if st.button("📸 Vérifier la Présence") and location:
                new_img = capture_image()
                if new_img is not None:
                    match = verifier_visage(path, new_img)
                    base_loc = st.session_state.base_location or location
                    distance = geodesic(base_loc, location).meters
                    st.image(new_img, caption="Image capturée", use_container_width=True)

                    if match:
                        if distance <= 100:
                            st.success(f"✅ Agent reconnu à {int(distance)} m : Présence validée.")
                            enregistrer_presence(tel, location, distance, "Validée")
                        else:
                            st.error(f"❌ Trop éloigné : {int(distance)} m > 100 m.")
                            enregistrer_presence(tel, location, distance, "Refusée - Trop éloigné")
                    else:
                        st.error("❌ Visage non reconnu.")
                        enregistrer_presence(tel, location, 0, "Refusée - Visage non reconnu")
                else:
                    st.error("Erreur de webcam.")
    else:
        st.warning("Numéro invalide (8 chiffres).")

# Journal
elif choice == "📊 Journal de Présence":
    st.subheader("📊 Journal de Présence des Agents")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)

        # Filtrage
        unique_tels = df["Telephone"].unique().tolist()
        tel_filter = st.selectbox("Filtrer par numéro de téléphone", ["Tous"] + unique_tels)

        if tel_filter != "Tous":
            df = df[df["Telephone"] == tel_filter]

        st.dataframe(df, use_container_width=True)

        # Téléchargement
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📤 Télécharger le journal CSV",
            data=csv,
            file_name="journal_presence.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucune donnée de présence enregistrée pour l’instant.")
