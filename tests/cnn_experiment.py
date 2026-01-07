#!/usr/bin/env python3
"""
ИСПРАВЛЕННЫЙ ЭКСПЕРИМЕНТ CNN - ПРАВИЛЬНЫЕ РАЗМЕРЫ
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Отключаем warning'и torchvision
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

print("="*80)
print("ЭКСПЕРИМЕНТ: CNN ДЛЯ РАСПОЗНАВАНИЯ ЛИЦ (LFW ДАТАСЕТ)")
print("="*80)

# Создаем директории
os.makedirs('results/cnn', exist_ok=True)

# ============================================================================
# 1. ДАННЫЕ
# ============================================================================
print("\n[1/6] ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")

start_time = time.time()
lfw_data = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_data.images
y = lfw_data.target
target_names = lfw_data.target_names

print(f"   ✓ Загружено: {len(X)} изображений")
print(f"   ✓ Классов: {len(target_names)}")
print(f"   ✓ Размер изображений: {X[0].shape}")
print(f"   ✓ Список людей: {', '.join(target_names)}")

# Нормализация и добавление размерности канала
X = X / 255.0  # [0, 1]
X = X[:, np.newaxis, :, :]  # (n_samples, 1, height, width)

print(f"   ✓ Формат данных: {X.shape}")

# Разделение
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"\n   Разделение данных:")
print(f"     Обучающая выборка:  {len(X_train):4d} изображений")
print(f"     Валидационная выборка: {len(X_val):4d} изображений")
print(f"     Тестовая выборка:    {len(X_test):4d} изображений")

# Конвертация в тензоры
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

data_load_time = time.time() - start_time
print(f"   ✓ Данные подготовлены за {data_load_time:.1f} секунд")

# ============================================================================
# 2. МОДЕЛЬ - ИСПРАВЛЕННАЯ АРХИТЕКТУРА
# ============================================================================
print("\n[2/6] СОЗДАНИЕ МОДЕЛИ CNN (исправленная архитектура)")

class EfficientFaceCNN(nn.Module):
    """Эффективная CNN для распознавания лиц - ИСПРАВЛЕННЫЕ РАЗМЕРЫ"""
    def __init__(self, num_classes, input_height=50, input_width=37, dropout_rate=0.3):
        super().__init__()
        
        # Вычисляем размеры после сверточных слоев
        # Начинаем с: 1 x 50 x 37
        
        # После conv1 + pool: 32 x 25 x 18 (50/2=25, 37/2=18.5 -> 18)
        # После conv2 + pool: 64 x 12 x 9 (25/2=12.5 -> 12, 18/2=9)
        # После conv3 + pool: 128 x 6 x 4 (12/2=6, 9/2=4.5 -> 4)
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 1x50x37 -> 32x25x18
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Block 2: 32x25x18 -> 64x12x9
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 3: 64x12x9 -> 128x6x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Вычисляем размер фичей после сверточных слоев
        # 128 каналов * 6 высота * 4 ширина = 3072
        self.feature_size = 128 * 6 * 4
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        print(f"     Размер входа: 1x{input_height}x{input_width}")
        print(f"     Размер фичей после conv слоев: {self.feature_size}")
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Создаем модель
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientFaceCNN(len(target_names)).to(device)

# Информация о модели
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   ✓ Архитектура: EfficientFaceCNN")
print(f"   ✓ Устройство: {device}")
print(f"   ✓ Всего параметров: {total_params:,}")
print(f"   ✓ Обучаемых параметров: {trainable_params:,}")

# ============================================================================
# 3. ОБУЧЕНИЕ
# ============================================================================
print("\n[3/6] НАСТРОЙКА ОБУЧЕНИЯ")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

batch_size = 16  # Уменьшили для стабильности
epochs = 10      # Уменьшили для быстрого теста

print(f"   ✓ Loss функция: CrossEntropyLoss")
print(f"   ✓ Оптимизатор: AdamW (lr=0.001)")
print(f"   ✓ Scheduler: CosineAnnealingLR")
print(f"   ✓ Batch size: {batch_size}")
print(f"   ✓ Эпох: {epochs}")

def get_batches(X, y, batch_size, shuffle=True):
    """Генератор батчей"""
    n_samples = len(X)
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        yield X[batch_idx], y[batch_idx]

# ============================================================================
# 4. ЦИКЛ ОБУЧЕНИЯ
# ============================================================================
print("\n[4/6] НАЧАЛО ОБУЧЕНИЯ")

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'lr': []
}

best_val_acc = 0.0
best_model_state = None

for epoch in range(epochs):
    epoch_start = time.time()
    
    # --- ОБУЧЕНИЕ ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    num_batches = 0
    
    for X_batch, y_batch in get_batches(X_train_t, y_train_t, batch_size, shuffle=True):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Проверяем размеры
        if num_batches == 0:
            print(f"     Размер батча: {X_batch.shape}")
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
        num_batches += 1
    
    train_loss = train_loss / num_batches if num_batches > 0 else 0
    train_acc = 100. * train_correct / train_total if train_total > 0 else 0
    
    # --- ВАЛИДАЦИЯ ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    num_val_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in get_batches(X_val_t, y_val_t, batch_size, shuffle=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()
            num_val_batches += 1
    
    val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
    val_acc = 100. * val_correct / val_total if val_total > 0 else 0
    
    # --- СОХРАНЕНИЕ ИСТОРИИ ---
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    # --- СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
    
    # --- ВЫВОД ИНФОРМАЦИИ ---
    epoch_time = time.time() - epoch_start
    
    print(f"\n   Эпоха {epoch+1:2d}/{epochs}:")
    print(f"     Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    print(f"     Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    print(f"     LR:    {optimizer.param_groups[0]['lr']:.6f}")
    print(f"     Время: {epoch_time:.1f} сек")
    
    if val_acc == best_val_acc:
        print(f"     ✅ Новая лучшая модель!")
    
    # --- ОБНОВЛЕНИЕ SCHEDULER ---
    scheduler.step()

# Загружаем лучшую модель
if best_model_state:
    model.load_state_dict(best_model_state)

print(f"\n   ✓ Обучение завершено!")
print(f"   ✓ Лучшая точность на валидации: {best_val_acc:.2f}%")

# ============================================================================
# 5. ТЕСТИРОВАНИЕ
# ============================================================================
print("\n[5/6] ТЕСТИРОВАНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ")

model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
all_predictions = []
all_true_labels = []
num_test_batches = 0

with torch.no_grad():
    for X_batch, y_batch in get_batches(X_test_t, y_test_t, batch_size, shuffle=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += y_batch.size(0)
        test_correct += (predicted == y_batch).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(y_batch.cpu().numpy())
        num_test_batches += 1

test_loss = test_loss / num_test_batches if num_test_batches > 0 else 0
test_acc = 100. * test_correct / test_total if test_total > 0 else 0

print(f"\n   РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
print(f"   {'='*40}")
print(f"     Test Loss:     {test_loss:.4f}")
print(f"     Test Accuracy: {test_acc:.2f}%")
print(f"     Correct/Total: {test_correct}/{test_total}")

# Отчет по классификации
print(f"\n   ОТЧЕТ ПО КЛАССИФИКАЦИИ:")
print(f"   {'='*40}")
if len(all_true_labels) > 0:
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=target_names, digits=3))
else:
    print("Нет данных для отчета")

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ И СОХРАНЕНИЕ
# ============================================================================
print("\n[6/6] ВИЗУАЛИЗАЦИЯ И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")

# 6.1. Графики обучения
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2, label='Train Loss')
axes[0, 0].plot(history['val_loss'], 'r-', linewidth=2, label='Val Loss')
axes[0, 0].set_title('Функция потерь', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Эпоха')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history['train_acc'], 'b-', linewidth=2, label='Train Accuracy')
axes[0, 1].plot(history['val_acc'], 'r-', linewidth=2, label='Val Accuracy')
axes[0, 1].axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.7, 
                   label=f'Best Val: {best_val_acc:.1f}%')
axes[0, 1].set_title('Точность', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Эпоха')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Learning Rate
axes[1, 0].plot(history['lr'], 'g-', linewidth=2, marker='o', markersize=4)
axes[1, 0].set_title('Скорость обучения', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Эпоха')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].grid(True, alpha=0.3)

# Матрица ошибок (если есть данные)
if len(all_true_labels) > 0:
    cm = confusion_matrix(all_true_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=target_names, yticklabels=target_names)
    axes[1, 1].set_title('Матрица ошибок', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Предсказанные метки')
    axes[1, 1].set_ylabel('Истинные метки')
else:
    axes[1, 1].text(0.5, 0.5, 'Нет данных\nдля матрицы ошибок', 
                    ha='center', va='center', fontsize=12)
    axes[1, 1].set_title('Матрица ошибок', fontsize=14, fontweight='bold')

plt.suptitle(f'CNN для распознавания лиц (LFW датасет)\n'
             f'Test Accuracy: {test_acc:.2f}% | Model: EfficientFaceCNN', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/cnn/training_results.png', dpi=150, bbox_inches='tight')

# 6.2. Примеры предсказаний
fig, axes = plt.subplots(2, 4, figsize=(15, 8))

model.eval()
with torch.no_grad():
    displayed = 0
    for i in range(min(8, len(X_test))):
        image = X_test[i].squeeze()
        true_label = y_test[i]
        
        # Предсказание
        input_tensor = torch.FloatTensor(X_test[i:i+1]).to(device)
        output = model(input_tensor)
        probs = torch.softmax(output, 1)
        pred_prob, pred_label = torch.max(probs, 1)
        
        ax = axes[i // 4, i % 4]
        ax.imshow(image, cmap='gray')
        
        true_name = target_names[true_label]
        pred_name = target_names[pred_label.item()]
        
        color = 'green' if true_label == pred_label.item() else 'red'
        border_color = color
        
        ax.set_title(f"True: {true_name}\nPred: {pred_name}\nProb: {pred_prob.item():.2f}", 
                     fontsize=9, color=color)
        ax.axis('off')
        
        # Добавляем рамку
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        
        displayed += 1
    
    # Заполняем оставшиеся пустые места
    for i in range(displayed, 8):
        ax = axes[i // 4, i % 4]
        ax.axis('off')

plt.suptitle('Примеры предсказаний модели', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/cnn/prediction_examples.png', dpi=150, bbox_inches='tight')

# 6.3. Сохранение модели и результатов
torch.save({
    'model_state_dict': model.state_dict(),
    'target_names': target_names,
    'test_accuracy': test_acc,
    'history': history,
    'model_config': {
        'num_classes': len(target_names),
        'total_params': total_params,
        'input_shape': X.shape[1:]
    }
}, 'results/cnn/face_recognition_model.pth')

# Сохранение метрик
import json
results_summary = {
    'experiment': 'CNN Face Recognition',
    'dataset': 'LFW',
    'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'device': str(device),
    'model': 'EfficientFaceCNN',
    'parameters': total_params,
    'best_val_accuracy': best_val_acc,
    'test_accuracy': test_acc,
    'test_loss': test_loss,
    'training_epochs': epochs,
    'batch_size': batch_size,
    'num_classes': len(target_names),
    'class_names': list(target_names),
    'sample_counts': {
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test)
    }
}

with open('results/cnn/experiment_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n   ✓ Результаты сохранены:")
print(f"     - Графики: results/cnn/training_results.png")
print(f"     - Примеры: results/cnn/prediction_examples.png")
print(f"     - Модель:  results/cnn/face_recognition_model.pth")
print(f"     - Метрики: results/cnn/experiment_summary.json")

# ============================================================================
# ФИНАЛЬНЫЙ ОТЧЕТ
# ============================================================================
print("\n" + "="*80)
print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН УСПЕШНО!")
print("="*80)
print(f"\n📊 РЕЗУЛЬТАТЫ:")
print(f"   {'─'*40}")
print(f"   │ Датасет:          LFW ({len(X)} изображений)")
print(f"   │ Классы:           {len(target_names)} человек")
print(f"   │ Архитектура:      EfficientFaceCNN")
print(f"   │ Параметры:        {total_params:,}")
print(f"   │ Устройство:       {device}")
print(f"   {'─'*40}")
print(f"   │ Лучшая Val Acc:   {best_val_acc:.2f}%")
print(f"   │ Test Accuracy:    {test_acc:.2f}%")
print(f"   │ Test Loss:        {test_loss:.4f}")
print(f"   {'─'*40}")

print(f"\n✅ ПЕРВЫЙ ЭКСПЕРИМЕНТ (CNN) ЗАВЕРШЕН!")
print("="*80)

plt.show()