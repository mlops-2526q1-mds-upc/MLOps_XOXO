import torch
import torch.nn as nn

class GenderAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )
        self.gender_head = nn.Linear(128, 2)
        self.age_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x = self.classifier(x)
        gender = self.gender_head(x)
        age = self.age_head(x)
        return torch.cat([gender, age], dim=1)


class GenderAgeLoss(nn.Module):
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