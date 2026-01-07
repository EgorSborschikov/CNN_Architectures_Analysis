import yaml
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class DataConfig:
    """Конфигурация данных"""
    dataset_name: str = "lfw"
    dataset_path: str = "./data/raw/lfw"
    min_faces_per_person: int = 70
    resize_factor: float = 0.4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config['datasets']['lfw'])
    
@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str, model_type: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config[model_type]
        return cls(**model_config['training'])
    
def load_config(config_path: str, model_type: str = None) -> Dict[str, Any]:
    """Загрузка конфигурации из YAML файла"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type and model_type in config:
        return config[model_type]
    return config