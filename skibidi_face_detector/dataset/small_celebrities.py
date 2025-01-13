import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from collections import Counter


dataset = load_dataset("theneuralmaze/celebrity_faces").with_format('torch')['train']
label_counts = Counter(dataset['label'])


def filter_minimum_n_examples(example):
    return label_counts[example['label']] >= 13


filtered_dataset = dataset.filter(filter_minimum_n_examples)

transformer = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
])


def transform(example):
    example['image'] = transformer(example['image'])
    return example


ds = filtered_dataset.map(transform, batched=False)

split = ds.train_test_split(test_size=0.2)
train_ds, test_ds = split['train'], split['test']

train_loader = DataLoader(train_ds, shuffle=True)
test_loader = DataLoader(test_ds, shuffle=False)

augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),
])
