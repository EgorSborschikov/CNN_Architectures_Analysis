import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import os

class CNNTrainer:
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 experiment_dir: str = "./experiments/cnn"):
        self.model = model.to(device)
        self.device = device
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": []
        }

    def train_epoch(self,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Статистика
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })

        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc
    
    def validate(self, 
                 val_loader: DataLoader,
                 criterion: nn.Module) -> Tuple[float, float, List[int], List[int]]:
        """Валидация"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = total_loss / total
        val_acc = 100. * correct / total
        
        return val_loss, val_acc, all_preds, all_labels
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              criterion: nn.Module,
              optimizer: optim.Optimizer,
              scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
              num_epochs: int = 30,
              patience: int = 10) -> Dict:
        """Полный цикл обучения"""
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(self.experiment_dir, "best_model.pth")
        
        print(f"Начинаю обучение на {self.device}...")
        print(f"Эпох: {num_epochs}, Размер батча: {train_loader.batch_size}")
        
        for epoch in range(num_epochs):
            print(f"\nЭпоха {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Обучение
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Валидация
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)
            
            # Сохраняем историю
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(
                optimizer.param_groups[0]['lr']
            )
            
            # Логируем
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Сохраняем лучшую модель
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, best_model_path)
                print(f"Лучшая модель сохранена! Accuracy: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch+1}")
                break
            
            # Обновляем scheduler
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        # Загружаем лучшую модель
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\nОбучение завершено!")
        print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")
        
        return self.history