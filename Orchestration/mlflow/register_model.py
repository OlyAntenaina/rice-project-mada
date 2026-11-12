import mlflow

# Remplace par ton Run ID et le nom de ton fichier modèle
run_id = "37741babc24443ea9be5a3c67be723ec"
model_name = "rice_model_MobileNetV2.h5"
registered_model_name = "Rice_Disease_Model"

# Construire l'URI du modèle (le chemin vers l'artefact dans le run)
model_uri = f"runs:/{run_id}/{model_name}"

# Enregistrer le modèle dans le registre
print(f"Enregistrement du modèle '{registered_model_name}' depuis {model_uri}...")
try:
    registered_model = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    print(f"✅ Modèle enregistré avec succès ! Version : {registered_model.version}")
except Exception as e:
    print(f"❌ Erreur lors de l'enregistrement : {e}")