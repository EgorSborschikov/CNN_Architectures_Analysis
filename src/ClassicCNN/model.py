import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List

class FaceClassifier(nn.Module):
    """Классическая CNN для классификации лиц"""
    
    def __init__(self, 
                 num_classes: int,
                 backbone: str = "resnet18",
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        super().__init__()
        
        # Словарь для загрузки весов
        weights_dict = {
            "resnet18": models.ResNet18_Weights.DEFAULT if pretrained else None,
            "resnet34": models.ResNet34_Weights.DEFAULT if pretrained else None,
            "resnet50": models.ResNet50_Weights.DEFAULT if pretrained else None,
            "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT if pretrained else None,
            "mobilenet_v2": models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        }
        
        if backbone not in weights_dict:
            raise ValueError(f"Unknown backbone: {backbone}. "
                           f"Available: {list(weights_dict.keys())}")
        
        # Загружаем модель с весами
        weights = weights_dict[backbone]
        
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet34":
            self.backbone = models.resnet34(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        # Создаем новый классификатор
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        
        # Статистика
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() 
                                    if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Конвертируем grayscale в RGB если нужно
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Получение эмбеддинга (для сравнения с сиамскими сетями)"""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        features = self.backbone(x)
        return features
    
    def get_model_info(self) -> dict:
        """Возвращает информацию о модели"""
        return {
            "total_parameters": self.total_params,
            "trainable_parameters": self.trainable_params,
            "architecture": str(self.backbone.__class__.__name__)
        }