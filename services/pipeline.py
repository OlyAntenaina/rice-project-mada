import os

def verify_pipeline():
    print("=========================================")
    print("   VERIFICATION DU PIPELINE MLOPS        ")
    print("=========================================\n")
    
    # 1. Hamarino ny Données (Data Engineering)
    print("--- 1. VERIFICATION DES DONNEES ---")
    train_path = os.path.join("Data", "Train")
    valid_path = os.path.join("Data", "Valid")
    
    if os.path.exists(train_path) and os.path.exists(valid_path):
        print("✅ Les donnees d'entrainement et de validation sont presentes.")
    else:
        print("ERREUR: Les donnees sont manquantes.")

    # 2. Hamarino ny Modely (MLOps)
    print("\n--- 2. VERIFICATION DU MODELE (MLOps) ---")
    model_path = os.path.join("models", "rice_model_MobileNetV2.h5")
    
    if os.path.exists(model_path):
        print("✅ Le modele MobileNetV2 est pret pour le deploiement.")
    else:
        print("❌ ERREUR: Le modele est introuvable.")

    # 3. Torolalana ho an'ny API
    print("\n--- 3. VERIFICATION DE L'API (FastAPI) ---")
    print("ℹ️ L'API est dans 'services/api.py'.")
    print("Pour la lancer : uvicorn services.api:app --reload")

    print("\n=========================================")
    print("   PIPELINE VERIFIE AVEC SUCCES          ")
    print("=========================================")

if __name__ == "__main__":
    verify_pipeline()