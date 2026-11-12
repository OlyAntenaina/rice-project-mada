import requests
import time
import os

# Ny lalana misy ny sary
image_folder = "Data/Test/Bacterial Blight/Field/"
api_url = "http://127.0.0.1:8000/predict"

print("🚁 Simulation Drone: Manomboka mandefa sary...")

# Maka sary 5 voalohany
try:
    images = os.listdir(image_folder)[:5]
except FileNotFoundError:
    print(f"Erreur: Tsy hita ny dossier {image_folder}. Jereo raha misy ny sary.")
    images = []

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    with open(img_path, "rb") as f:
        files = {"file": (img_name, f, "image/jpeg")}
        # Alefa miaraka amin'ny GPS (Simulation Antananarivo)
        data = {"latitude": -18.8792, "longitude": 47.5079}
        
        try:
            response = requests.post(api_url, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                print(f"📸 Drone nandefa: {img_name} | Valiny: {result['maladie']} ({result['confiance']*100:.1f}%)")
            else:
                print(f"❌ Erreur API: {response.status_code}")
        except Exception as e:
            print(f"❌ Tsy tafiditra ny fifandraisana: {e}")
    
    # Miandry 2 segondra toy ny drone manidina
    time.sleep(2)

print("simulation")