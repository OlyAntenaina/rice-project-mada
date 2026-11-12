import streamlit as st
import requests
import cv2
import tempfile
import numpy as np
from PIL import Image

st.set_page_config(page_title="Détection Maladies du Riz", layout="centered")
st.title("Famantarana ny Aretin'ny Vary (IA)")
st.write("Ampidiro sary na video hijery ny aretiny (Simulation Drone).")

# Safidy hisafidianana Sary na Video
option = st.radio("Safidio ny karazana fichier :", ("Sary (Image)", "Video (Drone)"))

if option == "Sary (Image)":
    uploaded_file = st.file_uploader("Misafidiana sary (JPG, PNG)", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Sary nampidirina", use_container_width=True)
        
        if st.button("Analyser l'image"):
            with st.spinner("Fandinihana..."):
                files = {"file": uploaded_file.getvalue()}
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Diagnostic : **{result['maladie']}**")
                    st.info(f"Confiance : {result['confiance'] * 100:.2f}%")
                    st.warning("Fanamarihana: Ny modely dia mifidy ny aretina manana isa ambony indrindra (Single-label).")

elif option == "Video (Drone)":
    uploaded_video = st.file_uploader("Misafidiana video (MP4)", type=["mp4", "avi"])
    
    if uploaded_video is not None:
        # Tehirizina vonjimaika ny video
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        st.video(tfile.name)
        
        if st.button("Analyser la vidéo (Simulation Drone)"):
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            st.write("Manomboka ny famakafakana...")
            st_frame = st.empty() # Hampisehoana ny sary
            results_area = st.empty() # Hampisehoana ny valiny
            
            results_list = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Isaky ny 1 segondra (fps) dia alefa any amin'ny API
                if frame_count % fps == 0:
                    _, buffer = cv2.imencode('.jpg', frame)
                    
                    with st.spinner("Fandinihana frame..."):
                        files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
                        try:
                            response = requests.post("http://127.0.0.1:8000/predict", files=files)
                            if response.status_code == 200:
                                res = response.json()
                                res_text = f"Segondra {frame_count//fps}s : **{res['maladie']}** ({res['confiance']*100:.1f}%)"
                                results_list.append(res_text)
                                
                                # Haseho ny frame sy ny valiny farany
                                st_frame.image(frame, channels="BGR", caption=f"Hita: {res['maladie']}")
                                results_area.write("\n".join(results_list[-5:])) # Haseho ny 5 farany
                        except Exception as e:
                            st.error(f"Erreur API: {e}")
                            
                frame_count += 1
                
            cap.release()
            st.success("succes")