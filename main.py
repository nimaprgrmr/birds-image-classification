import torch
import pandas as pd 
import numpy as np 
import torchvision 
from torchvision import transforms, datasets, models
from torch.utils.data import dataloader
import os

# birds_cv = pd.read_csv("birds_dataset/birds.csv")
# print(birds_cv.head())

# Becoming one with the data
def walk_through_dir(dir_path):
  """Walks through dir_path returning its contents."""
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


image_path = 'birds_dataset'
walk_through_dir(image_path)


# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir