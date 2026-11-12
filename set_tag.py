from mlflow import MlflowClient

# Mampifandray amin'ny serveur MLflow
client = MlflowClient()
model_name = "Rice_Disease_Model"
alias_name = "production"
version = 1

try:
    # Mametraka ny alias 'production' amin'ny version 1
    client.set_registered_model_alias(name=model_name, alias=alias_name, version=version)
    print(f"✅ Vita! Ny version {version} dia efa voatag ho '{alias_name}'.")
except Exception as e:
    print(f"❌ Nisy olana: {e}")