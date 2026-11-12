import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight

DATA_DIR = "Data" 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 
LEARNING_RATE = 0.001 

print("Loading data...")
train_ds = tf.keras.utils.image_dataset_from_directory(os.path.join(DATA_DIR, "Train"), image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(os.path.join(DATA_DIR, "Valid"), image_size=IMG_SIZE, batch_size=BATCH_SIZE)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# --- KAJY NY CLASS WEIGHTS (Fandanjalanjana ny Données) ---
y_train = np.concatenate([y for x, y in train_ds], axis=0)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

mlflow.set_experiment("Rice-Benchmark-Final")
model_name = "MobileNetV2"

print(f"\n--- Training: {model_name} ---")

# 1. Ny base_model dia avela hianatra (Fine-tuning)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True

model = tf.keras.Sequential([
    base_model, 
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

with mlflow.start_run(run_name=model_name):
    # 2. Fampiofanana (Nampiana ny class_weight)
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS,
        class_weight=class_weights_dict
    )
    
    # 3. Fandinihana ny valiny
    val_preds = model.predict(val_ds)
    y_pred = np.argmax(val_preds, axis=1)
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    mlflow.log_param("model_name", model_name)
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
    mlflow.log_metric("macro_f1_score", f1)
    
    # 4. Matrice de Confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"confusion_matrix_{model_name}.png")
    mlflow.log_artifact(f"confusion_matrix_{model_name}.png")
    plt.close()

    # 5. Fitehirizana
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save(f"models/rice_model_{model_name}.h5")
    mlflow.log_artifact(f"models/rice_model_{model_name}.h5")
    
    print(f"Finished training {model_name}.")