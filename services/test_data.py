import tensorflow as tf
import os

DATA_DIR = "Data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("1. Maka ny données...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "Train"), 
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE
)

print("2. Sokajy hita:", train_ds.class_names)

print("3. Mijery sary 1...")
for images, labels in train_ds.take(1):
    print(f"Endrik'ilay sary (shape): {images.shape}")
    print(f"Labels (isa): {labels.numpy()[:5]}")
    print(f"Isa ambony indrindra amin'ny sary: {tf.reduce_max(images).numpy()}")