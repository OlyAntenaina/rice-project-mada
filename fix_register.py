import mlflow
import mlflow.tensorflow
import tensorflow as tf

print("Mampiditra ny modely...")
# Maka ilay modely efa niofana
model = tf.keras.models.load_model("models/rice_model_MobileNetV2.h5")

# Mamorona andrana vaovao hanoratana ny modely amin'ny format marina
with mlflow.start_run() as run:
    print("Mandefa ny modely any amin'ny MLflow...")
    mlflow.tensorflow.log_model(model, artifact_path="model")
    run_id = run.info.run_id
    print(f"Vita! Run ID: {run_id}")

# Manao register an'ilay modely
print("Manao register an'ny modely...")
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "Rice_Disease_Model")
print("✅ Vita soa aman-tsara ny register!")
