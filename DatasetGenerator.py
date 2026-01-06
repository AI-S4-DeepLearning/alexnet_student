import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import numpy as np
from pathlib import Path


class DatasetGenerator:
    """
    Class for loading an ImageFolder dataset and generating
    train/validation/test datasets.

    Attributes:
        seed (int): Random seed used for deterministic splitting.
        data_dir (Path): Path to the dataset root directory.
        img_size (int): Target image size for resizing operations.
        full_dataset (ImageFolder): Complete dataset loaded from disk.
        classes (list[str]): List of class names.
        class_to_idx (dict): Mapping from class names to integer labels.
        num_classes (int): Number of distinct classes.
    """
    def __init__(self, seed: int, data_dir: Path, img_size: int):
        self.seed = seed
        self.data_dir = Path(data_dir)
        self.img_size = img_size        
        self.full_dataset = ImageFolder(root=self.data_dir)
        self.classes = self.full_dataset.classes
        self.class_to_idx = self.full_dataset.class_to_idx
        self.num_classes = len(self.classes)
    
    def get_balanced_splits(self, train_ratio: float, val_ratio: float, min_class_size: int):
        """
        Create train/validation/test splits with class balancing.

        Underrepresented classes in the training split can be augmented
        up to a minimum class size.

        Args:
            train_ratio (float): Fraction of the dataset assigned to training.
            val_ratio (float): Fraction assigned to validation.
            min_class_size (int): Minimum number of samples per class.

        Returns:
            tuple: A tuple containing the training, validation,
                and test datasets.
        """
        train_ds, val_ds, test_ds = self.get_basic_splits(train_ratio, val_ratio)
        
        train_dataset = BasicDataset(train_ds, self.img_size)
        val_dataset = BasicDataset(val_ds, self.img_size)
        test_dataset = BasicDataset(test_ds, self.img_size)
        
        return train_dataset, val_dataset, test_dataset
        
    def get_basic_splits(self, train_ratio: float, val_ratio: float):
        """
        Create train/validation/test splits without augmentation.

        Args:
            train_ratio (float): Fraction of the dataset assigned to training.
            val_ratio (float): Fraction assigned to validation.

        Returns:
            tuple: The train, validation, and test subsets produced by
                ``torch.utils.data.random_split``.
        """
        self.classes = self.full_dataset.classes
        self.num_classes = len(self.classes)
        
        total_size = len(self.full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            self.full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        return train_dataset, val_dataset, test_dataset
    
# OPDRACHT: Implementeer de klasse BalancedDataset en pas deze toe, om de data te balanceren.

    
class BasicDataset(Dataset):
    """
    Dataset wrapper that applies resizing, tensor conversion, and
    normalization to images without any augmentation.

    Attributes:
        base_dataset (Dataset): The underlying dataset providing (image, label) pairs.
        transform (callable): Transformation pipeline applied to each image.
    """
    def __init__(self, base_dataset: Dataset, img_size: int):
        self.base_dataset = base_dataset
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    
    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int):
        """
        Retrieve and preprocess a single dataset item.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple ``(image, label)``, where the image has been
            resized, converted to a tensor, and normalized.
        """
        img, label = self.base_dataset[idx]
        img = self.transform(img)
        
        return img, label