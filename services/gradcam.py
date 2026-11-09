import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

# 1. Mampiditra ny modely feno
model = tf.keras.models.load_model("models/rice_model_MobileNetV2.h5")

# 2. Maka ny base_model (MobileNetV2) ao anatiny
base_model = model.layers[0] 

# 3. Ny sary
img_path = "Data/Test/Bacterial Blight/Field/bacterial_leaf_blight261.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# 4. Grad-CAM amin'ny base_model
gradcam = Gradcam(base_model)
cam = gradcam(CategoricalScore(0), img_array)

# 5. Fampisehoana
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img)
plt.imshow(cam[0], cmap='jet', alpha=0.5)
plt.title("Grad-CAM")
plt.axis('off')
plt.show()