"""
Prépare le dataset UTKFace pour l'entraînement GenderAge
Version adaptée pour les fichiers .jpg.chip.jpg
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
from tqdm import tqdm
import re

def prepare_utkface(source_dir, output_dir='data/gender_age', test_size=0.2):
    """
    Parse UTKFace et crée la structure train/val
    
    Supporte les deux formats:
    - Format standard: [age]_[gender]_[race]_[date].jpg
    - Format chip: [age]_[gender]_[race]_[date].jpg.chip.jpg
    
    Args:
        source_dir: Dossier contenant les images UTKFace
        output_dir: Dossier de sortie (data/gender_age par défaut)
        test_size: Proportion du validation set (0.2 = 20%)
    """
    source = Path(source_dir).expanduser()
    output = Path(output_dir)
    
    if not source.exists():
        print(f"❌ Erreur: Le dossier {source} n'existe pas!")
        return
    
    print("🔍 Parsing des images UTKFace...")
    print(f"📁 Source: {source}")
    
    # Parser toutes les images (support .jpg et .jpg.chip.jpg)
    data = []
    errors = []
    
    # Chercher les deux types d'extensions
    image_files = list(source.glob('*.jpg')) + list(source.glob('*.jpg.chip.jpg'))
    
    if len(image_files) == 0:
        print(f"❌ Erreur: Aucune image trouvée dans {source}")
        print("Vérifiez le chemin du dataset.")
        return
    
    print(f"📊 {len(image_files)} fichiers trouvés")
    
    for img_path in tqdm(image_files, desc="Parsing"):
        try:
            # Nettoyer le nom de fichier pour extraire les métadonnées
            # Exemple: 100_0_0_20170112213500903.jpg.chip.jpg
            # On veut: 100_0_0_20170112213500903
            
            filename = img_path.name
            # Retirer toutes les extensions (.jpg, .chip.jpg, etc.)
            base_name = filename.replace('.jpg.chip.jpg', '').replace('.jpg', '')
            
            # Parser: age_gender_race_timestamp
            parts = base_name.split('_')
            
            if len(parts) < 2:
                errors.append(f"{filename}: format invalide")
                continue
            
            age = int(parts[0])
            gender = int(parts[1])
            
            # Filtrer les âges aberrants
            if not (0 <= age <= 100):
                errors.append(f"{filename}: âge aberrant {age}")
                continue
            
            # Vérifier que l'image est lisible
            img = Image.open(img_path)
            width, height = img.size
            img.verify()
            
            # Filtrer les images trop petites
            if width < 48 or height < 48:
                errors.append(f"{filename}: image trop petite {width}x{height}")
                continue
            
            data.append({
                'path': img_path,
                'age': age,
                'gender': gender,
                'original_name': filename
            })
            
        except ValueError as e:
            errors.append(f"{filename}: erreur de parsing - {str(e)}")
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    
    print(f"\n✅ {len(data)} images valides")
    if len(errors) > 0:
        print(f"⚠️  {len(errors)} images ignorées")
        # Afficher quelques erreurs pour debug
        print("Exemples d'erreurs:")
        for err in errors[:5]:
            print(f"  - {err}")
    
    if len(data) == 0:
        print("❌ Erreur: Aucune image valide trouvée!")
        print("\nVérifications:")
        print("1. Le dossier contient-il des images ?")
        print("2. Les noms suivent-ils le format age_gender_race_timestamp ?")
        return
    
    df = pd.DataFrame(data)
    
    # Statistiques détaillées
    print(f"\n{'='*60}")
    print(f"📊 STATISTIQUES DU DATASET")
    print(f"{'='*60}")
    print(f"Total images: {len(df)}")
    print(f"\nDistribution par genre:")
    print(f"  Femmes (0): {(df['gender']==0).sum():>5} ({(df['gender']==0).sum()/len(df)*100:.1f}%)")
    print(f"  Hommes (1): {(df['gender']==1).sum():>5} ({(df['gender']==1).sum()/len(df)*100:.1f}%)")
    print(f"\nStatistiques d'âge:")
    print(f"  Moyenne: {df['age'].mean():.1f} ans")
    print(f"  Médiane: {df['age'].median():.0f} ans")
    print(f"  Min/Max: {df['age'].min()}/{df['age'].max()} ans")
    print(f"\nDistribution par tranches d'âge:")
    age_bins = [0, 18, 30, 50, 70, 100]
    age_labels = ['0-17', '18-29', '30-49', '50-69', '70+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    print(df['age_group'].value_counts().sort_index().to_string())
    
    # Vérifier l'équilibre
    gender_ratio = (df['gender']==0).sum() / (df['gender']==1).sum()
    if 0.8 <= gender_ratio <= 1.2:
        print(f"\n✅ Dataset bien équilibré (ratio F/H: {gender_ratio:.2f})")
    else:
        print(f"\n⚠️  Dataset déséquilibré (ratio F/H: {gender_ratio:.2f})")
    
    # Split stratifié train/val
    print(f"\n{'='*60}")
    print(f"📂 CRÉATION DU SPLIT TRAIN/VAL")
    print(f"{'='*60}")
    print(f"Proportion: {int((1-test_size)*100)}% train / {int(test_size*100)}% validation")
    
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['gender'],  # Garder la même proportion H/F
        random_state=42
    )
    
    print(f"Train: {len(train_df)} images")
    print(f"Val:   {len(val_df)} images")
    
    # Créer la structure de dossiers et copier les images
    for split, split_df in [('train', train_df), ('val', val_df)]:
        split_dir = output / split / 'images'
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📋 Traitement du split '{split}'...")
        labels = []
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copie {split}"):
            # Nom standardisé (toujours .jpg, sans .chip)
            new_name = f"{idx:06d}.jpg"
            
            # Copier l'image
            try:
                shutil.copy(row['path'], split_dir / new_name)
                
                # Ajouter au CSV
                labels.append({
                    'image_name': new_name,
                    'gender': row['gender'],
                    'age': row['age']
                })
            except Exception as e:
                print(f"Erreur lors de la copie de {row['original_name']}: {e}")
        
        # Sauvegarder le CSV
        labels_df = pd.DataFrame(labels)
        csv_path = output / split / 'labels.csv'
        labels_df.to_csv(csv_path, index=False)
        
        print(f"  ✅ {len(labels)} images copiées")
        print(f"  ✅ CSV sauvegardé: {csv_path}")
        
        # Stats du split
        split_labels_df = pd.DataFrame(labels)
        print(f"  📊 Femmes: {(split_labels_df['gender']==0).sum()} | Hommes: {(split_labels_df['gender']==1).sum()}")
    
    print(f"\n{'='*60}")
    print(f"🎉 DATASET PRÉPARÉ AVEC SUCCÈS!")
    print(f"{'='*60}")
    print(f"📁 Emplacement: {output.absolute()}")
    print(f"\n📝 Structure créée:")
    print(f"  {output}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/     ({len(train_df)} images)")
    print(f"  │   └── labels.csv")
    print(f"  └── val/")
    print(f"      ├── images/     ({len(val_df)} images)")
    print(f"      └── labels.csv")
    
    print(f"\n🚀 PROCHAINE ÉTAPE:")
    print(f"  python train_models.py --mode train --epochs 10")
    print()

def verify_dataset(output_dir='data/gender_age'):
    """Vérifie que le dataset a été correctement préparé"""
    output = Path(output_dir)
    
    print("\n🔍 Vérification du dataset...")
    
    checks = []
    
    # Vérifier train
    train_images = output / 'train' / 'images'
    train_csv = output / 'train' / 'labels.csv'
    checks.append(('Train images', train_images.exists(), len(list(train_images.glob('*.jpg'))) if train_images.exists() else 0))
    checks.append(('Train CSV', train_csv.exists(), len(pd.read_csv(train_csv)) if train_csv.exists() else 0))
    
    # Vérifier val
    val_images = output / 'val' / 'images'
    val_csv = output / 'val' / 'labels.csv'
    checks.append(('Val images', val_images.exists(), len(list(val_images.glob('*.jpg'))) if val_images.exists() else 0))
    checks.append(('Val CSV', val_csv.exists(), len(pd.read_csv(val_csv)) if val_csv.exists() else 0))
    
    print(f"\n{'Item':<20} {'Status':<10} {'Count'}")
    print("-" * 50)
    for name, exists, count in checks:
        status = "✅" if exists else "❌"
        print(f"{name:<20} {status:<10} {count}")
    
    all_ok = all(check[1] for check in checks)
    if all_ok:
        print("\n✅ Le dataset est prêt pour l'entraînement!")
    else:
        print("\n❌ Problème détecté. Relancez prepare_utkface()")

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Préparation du dataset UTKFace pour GenderAge")
    print("Support des formats .jpg et .jpg.chip.jpg")
    print("="*60)
    print()
    
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        # Chemin détecté dans votre système
        source = "/media/mathys/fc77771b-dc3a-4012-a082-5c2462d8210d/facial-analysis/dataset/UTKFace"
    
    print(f"📁 Source: {source}")
    print(f"📁 Output: data/gender_age/")
    print()
    
    # Préparer le dataset
    prepare_utkface(source)
    
    # Vérifier le résultat
    verify_dataset()