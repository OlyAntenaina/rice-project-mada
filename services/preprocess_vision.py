import os
import random
from PIL import Image

def executer_pipeline_vision(dossier_source=".", dossier_sortie="./data_split", taille_cible=(224, 224), ratio_train=0.8):
    """
    Pipeline DataOps : Ingestion, nettoyage de qualité (Pillow) et split Train/Test.
    """
    classes = ["Healthy_Images", "Unhealthy_Images"]
    
    print("🧹 Démarrage du pipeline de nettoyage automatique (Data Quality)...")
    print(f"📁 Source : {os.path.abspath(dossier_source)}")
    print(f"📁 Sortie : {os.path.abspath(dossier_sortie)}\n")
    
    # Création de l'arborescence cible
    for split in ["train", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(dossier_sortie, split, cls), exist_ok=True)
            
    for nom_classe in classes:
        chemin_source_classe = os.path.join(dossier_source, nom_classe)
        
        if not os.path.exists(chemin_source_classe):
            print(f"❌ Erreur : Le dossier source '{nom_classe}' est introuvable.")
            continue
            
        # Sélection exclusive des fichiers d'images valides
        extensions_valides = ('.png', '.jpg', '.jpeg', '.bmp')
        toutes_images = [f for f in os.listdir(chemin_source_classe) if f.lower().endswith(extensions_valides)]
        
        random.shuffle(toutes_images)
        
        # Calcul de la répartition des données
        point_coupure = int(len(toutes_images) * ratio_train)
        splits = {
            "train": toutes_images[:point_coupure],
            "test": toutes_images[point_coupure:]
        }
        
        print(f"📊 Nettoyage automatique de la catégorie : [{nom_classe}]")
        
        # Traitement unitaire des images par partition
        for nom_split, liste_images in splits.items():
            succes = 0
            for nom_img in liste_images:
                chemin_src = os.path.join(chemin_source_classe, nom_img)
                chemin_dest = os.path.join(dossier_sortie, nom_split, nom_classe, nom_img)
                
                try:
                    # Contrôle Qualité & Standardisation
                    with Image.open(chemin_src) as img:
                        img.verify()
                    with Image.open(chemin_src) as img:
                        img_propre = img.convert("RGB").resize(taille_cible)
                        img_propre.save(chemin_dest, "JPEG")
                        succes += 1
                except Exception:
                    continue # Écarte les fichiers corrompus
                    
            print(f"   ➔ Images validées pour [{nom_split}] -> {succes} fichiers.")
        print()
        
    print("✅ Pipeline terminé avec succès. Les données sont prêtes pour l'entraînement.")

if __name__ == "__main__":
    random.seed(42) # Reproductibilité MLOps
    executer_pipeline_vision()

