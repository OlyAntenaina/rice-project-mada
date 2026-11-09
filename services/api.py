from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI(title="Rice Disease Detection API")

model = tf.keras.models.load_model("models/rice_model_MobileNetV2.h5")
class_names = ['Bacterial Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Smut', 'Tungro']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).resize((224, 224))
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    
    return {"maladie": predicted_class, "confiance": round(confidence, 4)}