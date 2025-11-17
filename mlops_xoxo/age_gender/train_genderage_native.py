"""
Entraînement GenderAge avec modèle PyTorch natif (pas de conversion ONNX)
Compatible avec batch_size > 1
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse


class GenderAgeDataset(Dataset):
    """Dataset pour Gender/Age"""
    
    def __init__(self, csv_path, images_dir, input_size=(112, 112)):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.input_size = input_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Charger l'image
        img_path = self.images_dir / row['image_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        
        # Normaliser
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Labels
        gender = torch.tensor(row['gender'], dtype=torch.long)
        age = torch.tensor(row['age'] / 100.0, dtype=torch.float32)
        
        return image, gender, age


class GenderAgeModel(nn.Module):
    """
    Modèle simple mais efficace pour Gender et Age
    Architecture inspirée de MobileNet
    """
    
    def __init__(self):
        super(GenderAgeModel, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Gender head (2 classes)
        self.gender_head = nn.Linear(128, 2)
        
        # Age head (regression)
        self.age_head = nn.Linear(128, 1)
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Shared classifier
        x = self.classifier(x)
        
        # Gender and age predictions
        gender = self.gender_head(x)
        age = self.age_head(x)
        
        # Concatenate outputs [batch, 3] : [gender_logit_0, gender_logit_1, age]
        output = torch.cat([gender, age], dim=1)
        
        return output


class GenderAgeLoss(nn.Module):
    """Loss combinée gender + age"""
    
    def __init__(self, gender_weight=1.0, age_weight=1.0):
        super().__init__()
        self.gender_loss = nn.CrossEntropyLoss()
        self.age_loss = nn.MSELoss()
        self.gender_weight = gender_weight
        self.age_weight = age_weight
    
    def forward(self, predictions, gender_target, age_target):
        gender_pred = predictions[:, :2]
        age_pred = predictions[:, 2]
        
        loss_gender = self.gender_loss(gender_pred, gender_target)
        loss_age = self.age_loss(age_pred, age_target)
        
        total_loss = self.gender_weight * loss_gender + self.age_weight * loss_age
        
        return total_loss, loss_gender, loss_age


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîne pour une epoch"""
    model.train()
    total_loss = 0
    correct_gender = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, genders, ages in pbar:
        images = images.to(device)
        genders = genders.to(device)
        ages = ages.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss, loss_g, loss_a = criterion(outputs, genders, ages)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculer l'accuracy pour le genre
        gender_pred = torch.argmax(outputs[:, :2], dim=1)
        correct_gender += (gender_pred == genders).sum().item()
        total_samples += genders.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'gender_acc': f'{correct_gender/total_samples:.2%}'
        })
    
    avg_loss = total_loss / len(dataloader)
    gender_acc = correct_gender / total_samples
    
    return avg_loss, gender_acc


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0
    correct_gender = 0
    total_samples = 0
    age_errors = []
    
    with torch.no_grad():
        for images, genders, ages in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            genders = genders.to(device)
            ages = ages.to(device)
            
            outputs = model(images)
            loss, _, _ = criterion(outputs, genders, ages)
            
            total_loss += loss.item()
            
            gender_pred = torch.argmax(outputs[:, :2], dim=1)
            correct_gender += (gender_pred == genders).sum().item()
            total_samples += genders.size(0)
            
            # Erreur d'âge
            age_pred = outputs[:, 2] * 100
            age_true = ages * 100
            age_errors.extend((age_pred - age_true).abs().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    gender_acc = correct_gender / total_samples
    mae_age = np.mean(age_errors)
    
    return avg_loss, gender_acc, mae_age


def train_model(train_loader, val_loader, epochs=50, device='cpu', save_dir='trained_models'):
    """Fonction principale d'entraînement"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Créer le modèle
    model = GenderAgeModel().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    criterion = GenderAgeLoss(gender_weight=1.0, age_weight=2.0)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_mae': []
    }
    
    print(f"\n{'='*60}")
    print(f"Début de l'entraînement sur {device}")
    print(f"Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation
        val_loss, val_acc, val_mae = validate(
            model, val_loader, criterion, device
        )
        
        # Mise à jour du learning rate
        scheduler.step(val_loss)
        
        # Sauvegarder l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_mae'].append(val_mae)
        
        # Afficher les résultats
        print(f"\nRésultats Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
        print(f"  Age MAE:    {val_mae:.2f} ans")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), 
                save_dir / 'best_genderage_native.pth'
            )
            print(f"  ✓ Meilleur modèle sauvegardé!")
        
        # Checkpoint périodique
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Sauvegarder l'historique
    pd.DataFrame(history).to_csv(save_dir / 'training_history.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Entraînement terminé!")
    print(f"Meilleur Val Loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")
    
    return model, history


def export_to_onnx(model_path, output_path, device='cpu'):
    """Exporte le modèle PyTorch vers ONNX"""
    
    print(f"\nExport du modèle vers ONNX: {output_path}")
    
    model = GenderAgeModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Modèle ONNX exporté avec succès!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='data/gender_age/train')
    parser.add_argument('--val-data', type=str, default='data/gender_age/val')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output-dir', type=str, default='trained_models')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Entraînement GenderAge - Modèle PyTorch natif")
    print("="*60)
    
    # Charger les données
    print(f"\nChargement des données...")
    train_dataset = GenderAgeDataset(
        csv_path=Path(args.train_data) / 'labels.csv',
        images_dir=Path(args.train_data) / 'images'
    )
    val_dataset = GenderAgeDataset(
        csv_path=Path(args.val_data) / 'labels.csv',
        images_dir=Path(args.val_data) / 'images'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"✓ Train: {len(train_dataset)} images")
    print(f"✓ Val: {len(val_dataset)} images")
    
    # Entraîner
    trained_model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=args.device,
        save_dir=args.output_dir
    )
    
    # Exporter vers ONNX
    print("\nExport vers ONNX...")
    export_to_onnx(
        model_path=Path(args.output_dir) / 'best_genderage_native.pth',
        output_path=Path(args.output_dir) / 'genderage_retrained.onnx',
        device=args.device
    )
    
    print(f"\n{'='*60}")
    print("✓ Pipeline complet terminé!")
    print(f"Modèle PyTorch: {args.output_dir}/best_genderage_native.pth")
    print(f"Modèle ONNX: {args.output_dir}/genderage_retrained.onnx")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


"""# Entraîner avec batch_size=32 (beaucoup plus rapide!)
python train_genderage_native.py \
  --train-data data/gender_age/train \
  --val-data data/gender_age/val \
  --epochs 20 \
  --batch-size 32 \
  --device cpu \
  --output-dir trained_models
  """