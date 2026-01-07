import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BaseFaceDataset(Dataset):
    """Базовый класс для датасетов лиц"""
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Если изображение в numpy array, конвертируем
        if isinstance(image, np.ndarray):
            # Для grayscale изображений добавляем размерность канала
            if len(image.shape) == 2:
                image = image[np.newaxis, ...]  # Добавляем размерность канала
            elif len(image.shape) == 3 and image.shape[0] == 3:
                # RGB изображение, оставляем как есть
                pass
            else:
                # Неизвестный формат, конвертируем в tensor
                image = torch.FloatTensor(image)
        
        if self.transform:
            image = self.transform(image)
        elif isinstance(image, np.ndarray):
            # Если нет трансформации, конвертируем в tensor
            image = torch.FloatTensor(image)
            
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_class_distribution(self):
        """Возвращает распределение по классам"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))
    
class LFWDataset(BaseFaceDataset):
    """LFW датасет через sklearn"""
    def __init__(self, min_faces_per_person=70, resize=0.4, transform=None):
        super().__init__(data_dir=None, transform=transform)

        print("Загрузка LFW датасета...")
        lfw_data = fetch_lfw_people(
            min_faces_per_person=min_faces_per_person,
            resize=resize,
            color=False  # Grayscale
        )
        
        self.images = lfw_data.images
        self.labels = lfw_data.target
        self.label_to_name = {i: name for i, name in enumerate(lfw_data.target_names)}
        self.name_to_label = {name: i for i, name in enumerate(lfw_data.target_names)}
        
        print(f"Загружено {len(self.images)} изображений")
        print(f"Количество классов: {len(self.label_to_name)}")

class CelebADataset(BaseFaceDataset):
    """CelebA датасет"""
    def __init__(self, data_dir: str, transform=None, max_samples_per_class=100):
        super().__init__(data_dir, transform)
        
        # Загрузка идентичностей
        identity_file = os.path.join(data_dir, "identity_CelebA.txt")
        identities = pd.read_csv(
            identity_file, 
            sep=" ", 
            header=None, 
            names=["file", "identity"]
        )
        
        # Ограничение количества на класс для баланса
        sampled_identities = identities.groupby('identity').head(max_samples_per_class)

        # Маппинг идентичностей в метки
        unique_ids = sampled_identities['identity'].unique()
        self.label_to_name = {i: str(id) for i, id in enumerate(unique_ids)}
        self.name_to_label = {str(id): i for i, id in enumerate(unique_ids)}

        # Загружаем изображения
        img_dir = os.path.join(data_dir, "img_align_celeba")
        print(f"Загрузка CelebA датасета из {img_dir}...")
        
        for _, row in sampled_identities.iterrows():
            img_path = os.path.join(img_dir, row['file'])
            try:
                image = Image.open(img_path).convert('L')  # Grayscale
                image = np.array(image)
                
                self.images.append(image)
                self.labels.append(self.name_to_label[str(row['identity'])])
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")
        
        print(f"Загружено {len(self.images)} изображений")
        print(f"Количество классов: {len(self.label_to_name)}")

def split_dataset(dataset: BaseFaceDataset, train_ratio=0.7, val_ratio=0.15, 
                  test_ratio=0.15, random_seed=42) -> Tuple[BaseFaceDataset, ...]:
    """Разделение датасета на train/val/test с сохранением распределения классов"""
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Преобразуем в numpy для удобства
    images = np.array(dataset.images)
    labels = np.array(dataset.labels)
    
    # Первое разделение: train и temp (val+test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, 
        train_size=train_ratio,
        stratify=labels,
        random_state=random_seed
    )
    
    # Второе разделение: val и test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels,
        train_size=val_test_ratio,
        stratify=temp_labels,
        random_state=random_seed
    )
    
    # Создаем датасеты
    train_dataset = type(dataset)(transform=dataset.transform)
    val_dataset = type(dataset)(transform=dataset.transform)
    test_dataset = type(dataset)(transform=dataset.transform)
    
    train_dataset.images = train_images
    train_dataset.labels = train_labels
    train_dataset.label_to_name = dataset.label_to_name
    train_dataset.name_to_label = dataset.name_to_label
    
    val_dataset.images = val_images
    val_dataset.labels = val_labels
    val_dataset.label_to_name = dataset.label_to_name
    val_dataset.name_to_label = dataset.name_to_label
    
    test_dataset.images = test_images
    test_dataset.labels = test_labels
    test_dataset.label_to_name = dataset.label_to_name
    test_dataset.name_to_label = dataset.name_to_label
    
    return train_dataset, val_dataset, test_dataset 