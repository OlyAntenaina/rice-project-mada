from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
import requests
from datetime import datetime
import os
import json
import matplotlib.cm as cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

app = FastAPI(title="Rice Disease Detection API")

# ==========================================
# FIAROVANA SY CORS (Mba tsy hisy olana amin'ny navigateur)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
SECRET_TOKEN = "Master-Mada-Rice-2026"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Token diso")
    return credentials.credentials

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username == "admin" and password == "admin":
        return {"token": SECRET_TOKEN}
    raise HTTPException(status_code=401, detail="Anarana na Mot de passe diso")

# ==========================================
# AI, GRAD-CAM, SY TAHIRY
# ==========================================
model = tf.keras.models.load_model("models/rice_model_MobileNetV2.h5")
class_names = ['Bacterial Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Smut', 'Tungro']

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

HISTORY_FILE = "history.json"
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

app.mount("/static", StaticFiles(directory="Front"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('Front/index.html')

# ==========================================
# PREDICTION
# ==========================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    latitude: float = Form(...), 
    longitude: float = Form(...),
    superficie: float = Form(...),
    mode: str = Form("standard"),
    token: str = Depends(verify_token)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB').resize((224, 224))
    img_array = np.array(image)
    
    img_array_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
    img_array_expanded = np.expand_dims(img_array_preprocessed, axis=0)
    
    predictions = model(img_array_expanded, training=False).numpy()
    
    # Multi-label simulation
    top_indices = np.argsort(predictions[0])[::-1]
    top_1_idx = top_indices[0]
    top_2_idx = top_indices[1]
    
    top_1_conf = float(predictions[0][top_1_idx])
    top_2_conf = float(predictions[0][top_2_idx])
    
    detected_diseases = []
    if top_1_conf > 0.50:
        detected_diseases.append(f"{class_names[top_1_idx]} ({top_1_conf*100:.1f}%)")
    if top_2_conf > 0.20:
        detected_diseases.append(f"{class_names[top_2_idx]} ({top_2_conf*100:.1f}%)")
        
    if top_1_conf < 0.60:
        predicted_class_text = "Tsy fantatra (Sary tsy mazava na tsy ravina)"
    else:
        predicted_class_text = " + ".join(detected_diseases)

    predicted_index = top_1_idx if top_1_conf >= 0.60 else -1
    confidence = top_1_conf

    # NDVI Simulation
    ndvi_text = "Tsy misy"
    if mode == "multispectral":
        red = img_array[:, :, 0].astype(float)
        green = img_array[:, :, 1].astype(float)
        ndvi = (green - red) / (green + red + 1e-5)
        mean_ndvi = np.mean(ndvi)
        if mean_ndvi < 0.15:
            ndvi_text = f"Risque Élevé (NDVI: {mean_ndvi:.2f}) - Infection précoce"
        else:
            ndvi_text = f"Indice Normal (NDVI: {mean_ndvi:.2f}) - Plante stable"

    # Grad-CAM
    gradcam_base64 = ""
    if predicted_index != -1:
        try:
            gradcam = Gradcam(model)
            cam = gradcam(CategoricalScore(predicted_index), img_array_expanded)
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            heatmap = np.array(Image.fromarray(heatmap).resize((224, 224)))
            superimposed = np.uint8(img_array * 0.5 + heatmap * 0.5)
            img_pil = Image.fromarray(superimposed)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG")
            gradcam_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print("Erreur GradCAM:", e)

    # Weather
    temp = "N/A"
    weather_code = 0
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        res = requests.get(url, timeout=3).json()
        weather = res.get('current_weather', {})
        temp = weather.get('temperature', 'N/A')
        weather_code = weather.get('weathercode', 0)
    except: pass

    # Save file
    file_extension = file.filename.split(".")[-1]
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    record = {
        "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "maladie": predicted_class_text,
        "confiance": round(confidence, 4),
        "temp": temp,
        "weather_code": weather_code,
        "media_url": f"/uploads/{filename}",
        "type": "video" if file_extension in ['mp4', 'avi', 'mov'] else "image",
        "localisation": {"latitude": latitude, "longitude": longitude},
        "superficie": superficie,
        "gradcam": gradcam_base64,
        "ndvi": ndvi_text
    }
    
    with open(HISTORY_FILE, "r+", encoding="utf-8") as f:
        history = json.load(f)
        history.insert(0, record)
        f.seek(0)
        json.dump(history, f, indent=4)

    return record

# ==========================================
# HISTORY
# ==========================================
@app.get("/history")
async def get_history():
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.delete("/history/{index}")
async def delete_history(index: int, token: str = Depends(verify_token)):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    if 0 <= index < len(history):
        item = history[index]
        file_name = item["media_url"].replace("/uploads/", "")
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        history.pop(index)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Tsy hita ilay tahiry")