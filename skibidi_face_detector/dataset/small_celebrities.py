import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from collections import Counter


ds = load_dataset("theneuralmaze/celebrity_faces").with_format('torch')['train']
label_counts = Counter(ds['label'])
labels_list = list(label_counts.keys())


def filter_minimum_n_examples(example):
    return label_counts[example['label']] >= 13


filtered_ds = ds.filter(filter_minimum_n_examples)

transformer = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
])


def transform(example):
    example['image'] = transformer(example['image'])
    example['label'] = labels_list.index(example['label'])
    return example


num_classes = len(set(filtered_ds['label']))
ds = filtered_ds.map(transform, batched=False)

split = ds.train_test_split(test_size=0.2)
train_ds, test_ds = split['train'], split['test']

train_loader = DataLoader(train_ds, shuffle=True)
test_loader = DataLoader(test_ds, shuffle=False)

augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
])
