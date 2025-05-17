import streamlit as st
import cv2
import numpy as np
import os
import time
from deepface import DeepFace
from geopy.distance import geodesic
from streamlit_javascript import st_javascript
from datetime import datetime
import pandas as pd

# Créer le dossier des photos s’il n’existe pas
if not os.path.exists("photos"):
    os.makedirs("photos")

log_path = "journal_presence.csv"

st.set_page_config(page_title="Contrôle de Présence", layout="centered")
st.title("📍 Application de Contrôle de Présence")

menu = ["📷 Enregistrement", "✅ Vérification", "📊 Journal de Présence"]
choice = st.sidebar.radio("Navigation", menu)

if "base_location" not in st.session_state:
    st.session_state.base_location = None

# Récupérer les coordonnées GPS via navigateur
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

# Capture image webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# Sauvegarder image agent
def save_image(img_rgb, tel):
    path = f"photos/{tel}.jpg"
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return path

# Enregistrer la présence
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

# === 📷 ENREGISTREMENT ===
if choice == "📷 Enregistrement":
    st.subheader("Étape 1 : Enregistrement de l'agent")
    tel = st.text_input("Numéro de téléphone de l'agent", max_chars=8)

    if len(tel) == 8 and tel.isdigit():
        location = get_real_location()
        if location:
            st.success(f"📌 Coordonnées GPS détectées : {location}")
            st.session_state.base_location = location
        else:
            st.warning("⚠️ GPS non disponible. Veuillez entrer manuellement.")
            lat = st.number_input("Latitude manuelle", value=6.1319, format="%.6f")
            lon = st.number_input("Longitude manuelle", value=1.2228, format="%.6f")
            location = (lat, lon)
            st.session_state.base_location = location

        if st.button("📸 Capturer et Enregistrer"):
            img = capture_image()
            if img is not None:
                save_image(img, tel)
                st.image(img, caption="Image capturée", use_container_width=True)
                st.success("✅ Agent enregistré avec succès.")
            else:
                st.error("Erreur lors de la capture d'image.")
    else:
        st.warning("Entrez un numéro valide (8 chiffres).")

# === ✅ VÉRIFICATION ===
elif choice == "✅ Vérification":
    st.subheader("Étape 2 : Vérification de la présence")
    tel = st.text_input("Numéro de téléphone de l'agent", max_chars=8)

    if len(tel) == 8 and tel.isdigit():
        ref_path = f"photos/{tel}.jpg"
        if not os.path.exists(ref_path):
            st.error("❌ Aucun enregistrement trouvé pour ce numéro.")
        else:
            location = get_real_location()
            if location:
                st.success(f"📌 Position actuelle : {location}")
            else:
                st.warning("⚠️ GPS non disponible. Veuillez entrer manuellement.")
                lat = st.number_input("Latitude actuelle", value=6.1319, format="%.6f")
                lon = st.number_input("Longitude actuelle", value=1.2228, format="%.6f")
                location = (lat, lon)

            if st.button("📸 Vérifier la Présence") and location:
                captured_img = capture_image()
                if captured_img is not None:
                    temp_img_path = f"photos/temp_{tel}.jpg"
                    cv2.imwrite(temp_img_path, cv2.cvtColor(captured_img, cv2.COLOR_RGB2BGR))

                    try:
                        result = DeepFace.verify(
                            img1_path=ref_path,
                            img2_path=temp_img_path,
                            enforce_detection=True
                        )
                        distance_m = geodesic(st.session_state.base_location, location).meters
                        st.image(captured_img, caption="Image capturée", use_container_width=True)

                        if result["verified"]:
                            if distance_m <= 100:
                                st.success(f"✅ Agent reconnu à {int(distance_m)} m : Présence validée.")
                                enregistrer_presence(tel, location, distance_m, "Validée")
                            else:
                                st.error(f"❌ Trop éloigné : {int(distance_m)} m > 100 m.")
                                enregistrer_presence(tel, location, distance_m, "Refusée - Trop éloigné")
                        else:
                            st.error("❌ Visage non reconnu.")
                            enregistrer_presence(tel, location, 0, "Refusée - Visage non reconnu")
                    except Exception as e:
                        st.error(f"Erreur DeepFace : {e}")
                else:
                    st.error("Erreur de capture via webcam.")
    else:
        st.warning("Entrez un numéro valide (8 chiffres).")

# === 📊 JOURNAL ===
elif choice == "📊 Journal de Présence":
    st.subheader("📊 Journal de Présence des Agents")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)

        tel_filter = st.selectbox("Filtrer par numéro de téléphone", ["Tous"] + df["Telephone"].unique().tolist())
        if tel_filter != "Tous":
            df = df[df["Telephone"] == tel_filter]

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📤 Télécharger le journal CSV", data=csv, file_name="journal_presence.csv", mime="text/csv")
    else:
        st.info("Aucune présence enregistrée.")
