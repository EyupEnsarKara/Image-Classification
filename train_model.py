import matplotlib.pyplot as plt
import torch
import torchvision
import os

from torch import nn
from torchvision import transforms
from utils import set_seeds
from engine_functions import train_model as train_engine

# --- Cell ---

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# --- Cell ---

# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Tüm parametreleri eğitilebilir yap
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = True
    
# 4. Change the classifier head 
class_names = sorted(os.listdir('yazlab-data/train'))  # Sınıfları alfabetik sıralı olarak al

set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
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
    # Önce PIL Image dönüşümleri
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1),  # Gri tonlama
    # ViT dönüşümleri (ToTensor içeriyor)
    pretrained_vit_transforms,
    # Tensor dönüşümleri (ToTensor'dan sonra gelmeli)
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))
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
                                                                                         transform=train_transforms,  # Eğitim için artırılmış veri
                                                                                         batch_size=8)

# --- Cell ---

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), 
                             lr=1e-4)  # Learning rate'i düşürdük
loss_fn = torch.nn.CrossEntropyLoss()

# Learning rate scheduler ekle
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3
)

# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()
pretrained_vit_results = train_engine(model=pretrained_vit,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      scheduler=scheduler,  # Scheduler'ı ekle
                                      epochs=15,
                                      device=device)

# --- Cell ---

# Plot the loss curves
from utils import plot_loss_curves

plot_loss_curves(pretrained_vit_results)

# Modeli kaydet
torch.save(pretrained_vit.state_dict(), 'animal_classifier_vit.pth')
print("Model kaydedildi: animal_classifier_vit.pth")

# --- Cell ---

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Confusion Matrix ve diğer metrikleri hesaplama fonksiyonu
def calculate_metrics(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred_logits = model(X)
            y_pred_labels = torch.argmax(y_pred_logits, dim=1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_pred_labels.cpu().numpy())
    
    # NumPy dizilerine dönüştür
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Metrikleri hesapla
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred
    }

# Test verisiyle metrikleri hesapla
test_metrics = calculate_metrics(pretrained_vit, test_dataloader_pretrained, device)

# Metrikleri yazdır
print(f"Precision (Kesinlik): {test_metrics['precision']:.4f}")
print(f"Recall (Duyarlılık): {test_metrics['recall']:.4f}")
print(f"F1-Score: {test_metrics['f1']:.4f}")

# Detaylı sınıflandırma raporu
print("\nSınıflandırma Raporu:")
print(classification_report(test_metrics['y_true'], test_metrics['y_pred'], target_names=class_names))

# --- Cell ---

# Confusion Matrix görselleştirme
plt.figure(figsize=(12, 10))
cm = test_metrics["confusion_matrix"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- Cell ---

# Öğrenme eğrileri ile birlikte metrik görselleştirme
plt.figure(figsize=(15, 10))

# 1. Loss eğrisi
plt.subplot(2, 2, 1)
plt.plot(pretrained_vit_results["train_loss"], label="Eğitim Kaybı")
plt.plot(pretrained_vit_results["test_loss"], label="Doğrulama Kaybı")
plt.title("Kayıp Eğrisi")
plt.xlabel("Epoch")
plt.ylabel("Kayıp")
plt.legend()

# 2. Doğruluk eğrisi
plt.subplot(2, 2, 2)
plt.plot(pretrained_vit_results["train_acc"], label="Eğitim Doğruluğu")
plt.plot(pretrained_vit_results["test_acc"], label="Doğrulama Doğruluğu")
plt.title("Doğruluk Eğrisi")
plt.xlabel("Epoch")
plt.ylabel("Doğruluk (%)")
plt.legend()

# 3. Precision, Recall, F1 skorlarını gösteren çubuk grafik
plt.subplot(2, 2, 3)
metrics = ["Precision", "Recall", "F1-Score"]
values = [test_metrics["precision"], test_metrics["recall"], test_metrics["f1"]]
bars = plt.bar(metrics, values, color=["#3498db", "#2ecc71", "#e74c3c"])
plt.title("Değerlendirme Metrikleri")
plt.ylim(0, 1.0)
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# 4. Sınıf başına düşen performans (precision for each class)
plt.subplot(2, 2, 4)
# Sınıf başına precision değerlerini hesapla
class_precision = precision_score(test_metrics["y_true"], test_metrics["y_pred"], average=None)
plt.bar(range(len(class_names)), class_precision, color="#9b59b6")
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.title("Sınıf Başına Kesinlik (Precision)")
plt.ylim(0, 1.0)
plt.tight_layout()

plt.savefig("model_metrics.png")
plt.show()

# Tüm metrikleri bir JSON dosyasına kaydet
import json
all_metrics = {
    "precision": float(test_metrics["precision"]),
    "recall": float(test_metrics["recall"]),
    "f1": float(test_metrics["f1"]),
    "accuracy": float(pretrained_vit_results["test_acc"][-1])
}

with open("model_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)

print("Metrikler kaydedildi: model_metrics.json")


