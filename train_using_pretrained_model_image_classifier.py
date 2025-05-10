import matplotlib.pyplot as plt
import torch
import torchvision
import os

from torch import nn
from torchvision import transforms
from helper_functions import set_seeds

# --- Cell ---

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# --- Cell ---

# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False
    
# 4. Change the classifier head 
class_names = sorted(os.listdir('yazlab-data/train'))

set_seeds()
# Daha karmaşık bir classifier head
class CustomHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_features)
        )
    
    def forward(self, x):
        return self.classifier(x)

pretrained_vit.heads = CustomHead(in_features=768, out_features=len(class_names)).to(device)
# pretrained_vit # uncomment for model output

# --- Cell ---

from torchinfo import summary

# Print a summary using torchinfo (uncomment for actual output)
summary(model=pretrained_vit, 
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# --- Cell ---

# Setup directory paths to train and test images
train_dir = 'yazlab-data/train'
test_dir = 'yazlab-data/val'

# --- Cell ---

# Get automatic transforms from pretrained ViT weights
pretrained_vit_transforms = pretrained_vit_weights.transforms()

# Veri artırma için ek dönüşümler
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    pretrained_vit_transforms
])

test_transforms = pretrained_vit_transforms

# --- Cell ---

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=False,  # CPU için pin_memory'yi kapattık
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=False,  # CPU için pin_memory'yi kapattık
  )

  return train_dataloader, test_dataloader, class_names

# --- Cell ---

# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                         test_dir=test_dir,
                                                                                         transform=train_transforms,
                                                                                         batch_size=4)  # Batch size'ı 8'den 4'e düşürdük

# --- Cell ---

from going_modular.going_modular import engine

# Create optimizer and loss function
optimizer = torch.optim.AdamW(params=pretrained_vit.parameters(), 
                            lr=5e-5,  # Learning rate'i düşürdük çünkü daha küçük batch size kullanıyoruz
                            weight_decay=0.01)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,  # İlk restart periyodu
    T_mult=2,  # Her restart sonrası periyodu 2 katına çıkar
    eta_min=1e-6  # Minimum learning rate
)

# Gradient clipping için
torch.nn.utils.clip_grad_norm_(pretrained_vit.parameters(), max_norm=1.0)

# Early stopping için
from torch.optim.lr_scheduler import ReduceLROnPlateau
early_stopping_patience = 5
best_loss = float('inf')
patience_counter = 0

# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()
pretrained_vit_results = engine.train(model=pretrained_vit,
                                    train_dataloader=train_dataloader_pretrained,
                                    test_dataloader=test_dataloader_pretrained,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    scheduler=scheduler,
                                    epochs=30,  # Daha fazla epoch
                                    device=device)

# --- Cell ---

# Plot the loss curves
from helper_functions import plot_loss_curves

plot_loss_curves(pretrained_vit_results)

# Modeli kaydet
torch.save(pretrained_vit.state_dict(), 'animal_classifier_vit.pth')
print("Model kaydedildi: animal_classifier_vit.pth")

# --- Cell ---

import requests

# Import function to make predictions on images and plot them 
from going_modular.going_modular.predictions import pred_and_plot_image

# Setup custom image path
custom_image_path = "test_img.jpg"

# Predict on custom image
pred_and_plot_image(model=pretrained_vit,
                    image_path=custom_image_path,
                    class_names=class_names)

# --- Cell ---

# Import function to make predictions on images and plot them 
from going_modular.going_modular.predictions import pred_and_plot_image

# Setup custom image path
custom_image_path = "test_1.jpg"

# Predict on custom image
pred_and_plot_image(model=pretrained_vit,
                    image_path=custom_image_path,
                    class_names=class_names)

# --- Cell ---

