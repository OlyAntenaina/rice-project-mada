import cv2
import os

# Ny dossier misy ny sary (ovay raha tianao)
image_folder = "Data/Test/Bacterial Blight/Field/" 
video_name = "simulation_drone.mp4"

# Maka ny anaran'ny sary rehetra
images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]

if not images:
    print("Tsy misy sary hita ao amin'ny dossier.")
else:
    # Maka ny refy (dimensions) an'ny sary voalohany
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Mamorona ny video (fps=1 dia 1 sary isan-tsegondra)
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    # Mampiditra sary 15 voalohany ao anaty video
    for image in images[:15]: 
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    print(f"✅ Vita ny video: {video_name}. Ampiasao amin'ny Streamlit izao!")