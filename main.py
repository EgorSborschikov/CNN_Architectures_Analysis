#!/usr/bin/env python3
"""
Главный скрипт для сравнения архитектур распознавания лиц
"""

import os
import sys
import json
import yaml
import argparse
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.data_loader import LFWDataset, CelebADataset, split_dataset
from src.utils.config import load_config
from src.ClassicCNN.model import FaceClassifier
from src.ClassicCNN.trainer import CNNTrainer

class ExperimentRunner:
    """Запуск и сравнение экспериментов"""
    
    def __init__(self, config_path: str = "./configs"):
        self.config_path = config_path
        self.results = {}
        
        # Создаем директории
        os.makedirs("./experiments", exist_ok=True)
        os.makedirs("./models", exist_ok=True)
    
    def setup_data(self, dataset_name: str = "lfw"):
        """Подготовка данных"""
        
        # Трансформации
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Загрузка датасета
        if dataset_name == "lfw":
            dataset = LFWDataset(
                min_faces_per_person=70,
                resize=0.4,
                transform=train_transform  # Для всего датасета
            )
        elif dataset_name == "celeba":
            dataset = CelebADataset(
                data_dir="./data/raw/celeba",
                transform=train_transform,
                max_samples_per_class=100
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Разделение
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        # Для валидации и теста используем val_transform
        val_dataset.transform = val_transform
        test_dataset.transform = val_transform
        
        # DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        
        print(f"\nДанные подготовлены:")
        print(f"  Обучающая выборка: {len(train_dataset)}")
        print(f"  Валидационная выборка: {len(val_dataset)}")
        print(f"  Тестовая выборка: {len(test_dataset)}")
        print(f"  Количество классов: {len(dataset.label_to_name)}")
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
            "num_classes": len(dataset.label_to_name),
            "class_names": dataset.label_to_name
        }
    
    def run_cnn_experiment(self, data_loaders: Dict, config: Dict):
        """Запуск эксперимента с классической CNN"""
        
        print("\n" + "="*60)
        print("ЭКСПЕРИМЕНТ: КЛАССИЧЕСКАЯ CNN")
        print("="*60)
        
        # Устройство
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Устройство: {device}")
        
        # Модель
        model = FaceClassifier(
            num_classes=data_loaders["num_classes"],
            backbone=config["model"]["backbone"],
            pretrained=config["model"]["pretrained"],
            dropout_rate=config["model"]["dropout_rate"]
        )
        
        model_info = model.get_model_info()
        print(f"\nИнформация о модели:")
        print(f"  Архитектура: {model_info['architecture']}")
        print(f"  Всего параметров: {model_info['total_parameters']:,}")
        print(f"  Обучаемых параметров: {model_info['trainable_parameters']:,}")
        
        # Критерий и оптимизатор
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Тренер
        trainer = CNNTrainer(
            model=model,
            device=device,
            experiment_dir="./experiments/cnn"
        )
        
        # Обучение
        history = trainer.train(
            train_loader=data_loaders["train"],
            val_loader=data_loaders["val"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config["training"]["num_epochs"],
            patience=config["training"]["early_stopping"]["patience"]
        )
        
        # Тестирование
        test_loss, test_acc, test_preds, test_labels = trainer.validate(
            data_loaders["test"], criterion
        )
        
        print(f"\nРезультаты на тестовой выборке:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        
        # Сохраняем результаты
        self.results["cnn"] = {
            "model_info": model_info,
            "history": history,
            "test_metrics": {
                "loss": test_loss,
                "accuracy": test_acc
            },
            "config": config
        }
        
        # Визуализация
        self.plot_cnn_results(history)
        
        return history, test_acc
    
    def plot_cnn_results(self, history: Dict):
        """Визуализация результатов CNN"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('Функция потерь', fontsize=14)
        axes[0].set_xlabel('Эпоха', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_title('Точность', fontsize=14)
        axes[1].set_xlabel('Эпоха', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./experiments/cnn_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Сохранение всех результатов"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"./experiments/comparison_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nРезультаты сохранены в: {results_file}")
        
        # Создаем таблицу сравнения
        comparison_data = []
        
        for model_name, result in self.results.items():
            comparison_data.append({
                "Model": model_name.upper(),
                "Backbone": result["model_info"]["architecture"],
                "Parameters": f"{result['model_info']['total_parameters']:,}",
                "Trainable": f"{result['model_info']['trainable_parameters']:,}",
                "Best Val Acc": f"{max(result['history']['val_acc']):.2f}%",
                "Test Acc": f"{result['test_metrics']['accuracy']:.2f}%",
                "Test Loss": f"{result['test_metrics']['loss']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print("\n" + "="*80)
        print("ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
        print("="*80)
        print(df.to_string(index=False))
        
        # Сохраняем таблицу
        csv_file = f"./experiments/comparison_table_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nТаблица сохранена в: {csv_file}")
        
        return df
    
    def run_all_experiments(self):
        """Запуск всех экспериментов"""
        
        # Загружаем конфигурации
        cnn_config = load_config("./configs/cnn_config.yaml", "cnn")
        
        # Подготовка данных
        print("="*60)
        print("ПОДГОТОВКА ДАННЫХ")
        print("="*60)
        
        data_loaders = self.setup_data(dataset_name="lfw")
        
        # Запуск CNN эксперимента
        self.run_cnn_experiment(data_loaders, cnn_config)
        
        # TODO: Добавить сиамские и триплетные сети
        print("\n" + "="*60)
        print("СИАМСКИЕ И ТРИПЛЕТНЫЕ СЕТИ БУДУТ ДОБАВЛЕНЫ В СЛЕДУЮЩЕЙ ВЕРСИИ")
        print("="*60)
        
        # Сохранение результатов
        self.save_results()

def main():
    parser = argparse.ArgumentParser(description="Сравнение архитектур распознавания лиц")
    parser.add_argument("--dataset", type=str, default="lfw",
                       choices=["lfw", "celeba"],
                       help="Датасет для использования")
    parser.add_argument("--model", type=str, default="all",
                       choices=["cnn", "siamese", "triplet", "all"],
                       help="Модель для обучения")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    runner.run_all_experiments()

if __name__ == "__main__":
    main()