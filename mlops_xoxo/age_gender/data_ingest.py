#!/usr/bin/env python3
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
import yaml

with open("pipelines/age_gender/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)


def prepare_utkface(source_dir, output_dir='data/processed/age_gender', test_size=0.2):
    """
    Parse UTKFace et crée la structure train/val
    
    Supporte les deux formats:
    - Format standard: [age]_[gender]_[race]_[date].jpg
    - Format chip: [age]_[gender]_[race]_[date].jpg.chip.jpg
    """
    source = Path(source_dir).expanduser()
    output = Path(output_dir)
    
    if not source.exists():
        print(f"❌ Erreur: Le dossier {source} n'existe pas!")
        return
    
    print("🔍 Parsing des images UTKFace...")
    print(f"📁 Source: {source}")
    
    data = []
    errors = []
    image_files = list(source.glob('*.jpg')) + list(source.glob('*.jpg.chip.jpg'))
    
    if len(image_files) == 0:
        print(f"❌ Aucune image trouvée dans {source}")
        return
    
    print(f"📊 {len(image_files)} fichiers trouvés")
    
    for img_path in tqdm(image_files, desc="Parsing"):
        try:
            filename = img_path.name
            base_name = filename.replace('.jpg.chip.jpg', '').replace('.jpg', '')
            parts = base_name.split('_')
            
            if len(parts) < 2:
                errors.append(f"{filename}: format invalide")
                continue
            
            age = int(parts[0])
            gender = int(parts[1])
            if not (0 <= age <= 100):
                continue
            
            img = Image.open(img_path)
            w, h = img.size
            img.verify()
            if w < 48 or h < 48:
                continue
            
            data.append({'path': img_path, 'age': age, 'gender': gender, 'original_name': filename})
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    
    df = pd.DataFrame(data)
    if len(df) == 0:
        print("❌ Erreur: aucune image valide trouvée!")
        return
    
    print(f"\n✅ {len(df)} images valides")
    print(f"⚠️ {len(errors)} erreurs ignorées")
    
    print("\n📊 STATISTIQUES DU DATASET")
    print("=" * 60)
    print(f"Total: {len(df)}")
    print(f"Femmes: {(df['gender']==0).sum()} | Hommes: {(df['gender']==1).sum()}")
    print(f"Âge moyen: {df['age'].mean():.1f} | Médiane: {df['age'].median():.0f}")
    
    # Age groups
    age_bins = [0, 18, 30, 50, 70, 100]
    age_labels = ['0-17', '18-29', '30-49', '50-69', '70+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    print(df['age_group'].value_counts().sort_index())
    
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df['gender'], random_state=42
    )
    
    for split, split_df in [('train', train_df), ('val', val_df)]:
        split_dir = output / split / 'images'
        split_dir.mkdir(parents=True, exist_ok=True)
        labels = []
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copie {split}"):
            new_name = f"{idx:06d}.jpg"
            shutil.copy(row['path'], split_dir / new_name)
            labels.append({'image_name': new_name, 'gender': row['gender'], 'age': row['age']})
        pd.DataFrame(labels).to_csv(output / split / 'labels.csv', index=False)
    
    print(f"\n🎉 Dataset prêt sous {output.absolute()}")


if __name__ == "__main__":
    src = params["dataset"]["raw_dir"]
    prepare_utkface(src)