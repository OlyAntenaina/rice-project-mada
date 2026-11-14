import cv2
import os

image_folder = "Data/Test/Bacterial Blight/Field/" 
video_name = "simulation_drone_h264.mp4"

images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]

if images:
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Ny fanovana dia eto: mp4v -> avc1
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

    for image in images[:15]:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    print(f"✅ Vita ny video: {video_name}")