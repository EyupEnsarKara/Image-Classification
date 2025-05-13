import matplotlib.pyplot as plt
import torch
import torchvision
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

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
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)), # Gaussian Blur eklendi
    transforms.RandomEqualize(p=0.5), # Rastgele
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
                                                                                         transform=train_transforms,  # Eğitim için artırılmış veri
                                                                                         batch_size=8)

# --- Cell ---

# Modeli değerlendirme fonksiyonu
def evaluate_model(model, dataloader, loss_fn, device):
    """
    Verilen model ve dataloader ile değerlendirme yapan fonksiyon.
    Accuracy, precision, recall ve F1 skoru hesaplar.
    """
    model.eval()
    
    # İzleme değişkenleri
    running_loss = 0.0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            running_loss += loss.item()
            
            # Tahminleri hesapla
            y_pred = torch.argmax(y_logits, dim=1)
            
            # CPU'ya al ve liste olarak sakla
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())
    
    # Metrikleri hesapla
    accuracy = np.mean(np.array(all_y_true) == np.array(all_y_pred))
    precision = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    
    # Ortalama kaybı hesapla
    avg_loss = running_loss / len(dataloader)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": all_y_true,
        "y_pred": all_y_pred
    }

# Gelişmiş learning curve çizim fonksiyonu
def plot_training_curves(results):
    """
    Eğitim sürecindeki metrik eğrilerini çizer.
    """
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss curves
    axs[0, 0].plot(results["train_loss"], label="Eğitim Kaybı")
    axs[0, 0].plot(results["test_loss"], label="Doğrulama Kaybı")
    axs[0, 0].set_title("Kayıp Eğrisi")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Kayıp")
    axs[0, 0].legend()
    
    # Plot accuracy curves
    axs[0, 1].plot(results["train_acc"], label="Eğitim Doğruluğu")
    axs[0, 1].plot(results["test_acc"], label="Doğrulama Doğruluğu")
    axs[0, 1].set_title("Doğruluk Eğrisi")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Doğruluk")
    axs[0, 1].legend()
    
    # Plot precision and recall for validation
    axs[1, 0].plot(results["test_precision"], label="Doğrulama Kesinliği")
    axs[1, 0].plot(results["test_recall"], label="Doğrulama Duyarlılığı")
    axs[1, 0].set_title("Kesinlik ve Duyarlılık Eğrisi")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Değer")
    axs[1, 0].legend()
    
    # Plot F1-Score
    axs[1, 1].plot(results["test_f1"], label="Doğrulama F1 Skoru")
    axs[1, 1].set_title("F1 Skoru Eğrisi")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("F1 Skoru")
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model_metrics.png')
    plt.show()

# Konfüzyon matrisi oluşturma ve görselleştirme
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Konfüzyon matrisini oluşturur ve görselleştirir.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Konfüzyon Matrisi')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# Tüm metrikleri bir tablo halinde gösterme
def display_metrics_table(train_metrics, test_metrics):
    """
    Eğitim ve test metriklerini bir tablo halinde gösterir.
    """
    metrics_df = pd.DataFrame({
        'Metrik': ['Doğruluk (Accuracy)', 'Kesinlik (Precision)', 'Duyarlılık (Recall)', 'F1 Skoru'],
        'Eğitim': [train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1']],
        'Test': [test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'], test_metrics['f1']]
    })
    
    # Sayısal değerleri yüzde olarak formatlama
    for col in ['Eğitim', 'Test']:
        metrics_df[col] = metrics_df[col].apply(lambda x: f'{x:.2%}')
    
    print("Performans Metrikleri Tablosu:")
    print(metrics_df.to_string(index=False))
    
    # Tabloyu görselleştirme
    plt.figure(figsize=(8, 4))
    table = plt.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.axis('off')
    plt.title('Performans Metrikleri Tablosu', fontsize=14)
    plt.tight_layout()
    plt.savefig('metrics_table.png')
    plt.show()

# --- Cell ---

from going_modular.going_modular import engine

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

# Geliştirilmiş train fonksiyonu
def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, scheduler, epochs, device):
    """
    Modeli eğiten ve gelişmiş metrikler döndüren fonksiyon.
    """
    # İzleme değişkenleri
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": []
    }
    
    # Eğitim döngüsü
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        
        # Eğitim moduna geç
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        all_train_y_true = []
        all_train_y_pred = []
        
        # Batch'ler üzerinde eğitim
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrikleri takip et
            train_loss += loss.item()
            y_pred = torch.argmax(y_logits, dim=1)
            train_correct += (y_pred == y).sum().item()
            train_total += y.size(0)
            
            # Metrikler için tahminleri sakla
            all_train_y_true.extend(y.cpu().numpy())
            all_train_y_pred.extend(y_pred.cpu().numpy())
            
            # İlerlemeyi göster
            if batch % 10 == 0:
                print(f"Batch: {batch}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
        
        # Epoch'un sonunda eğitim metriklerini hesapla
        train_accuracy = train_correct / train_total
        train_precision = precision_score(all_train_y_true, all_train_y_pred, average='macro', zero_division=0)
        train_recall = recall_score(all_train_y_true, all_train_y_pred, average='macro', zero_division=0)
        train_f1 = f1_score(all_train_y_true, all_train_y_pred, average='macro', zero_division=0)
        
        # Doğrulama metriklerini hesapla
        test_metrics = evaluate_model(model, test_dataloader, loss_fn, device)
        
        # Scheduler'ı güncelle
        scheduler.step(test_metrics["loss"])
        
        # Sonuçları kaydet
        results["train_loss"].append(train_loss / len(train_dataloader))
        results["train_acc"].append(train_accuracy)
        results["train_precision"].append(train_precision)
        results["train_recall"].append(train_recall)
        results["train_f1"].append(train_f1)
        
        results["test_loss"].append(test_metrics["loss"])
        results["test_acc"].append(test_metrics["accuracy"])
        results["test_precision"].append(test_metrics["precision"])
        results["test_recall"].append(test_metrics["recall"])
        results["test_f1"].append(test_metrics["f1"])
        
        # Epoch sonuçlarını yazdır
        print(f"Epoch: {epoch+1}/{epochs} | "
              f"Train Loss: {results['train_loss'][-1]:.4f} | "
              f"Train Acc: {results['train_acc'][-1]:.4f} | "
              f"Test Loss: {results['test_loss'][-1]:.4f} | "
              f"Test Acc: {results['test_acc'][-1]:.4f}")
    
    # Son test metrikleri
    final_train_metrics = {
        "loss": results["train_loss"][-1],
        "accuracy": results["train_acc"][-1],
        "precision": results["train_precision"][-1],
        "recall": results["train_recall"][-1],
        "f1": results["train_f1"][-1]
    }
    
    final_test_metrics = {
        "loss": results["test_loss"][-1],
        "accuracy": results["test_acc"][-1],
        "precision": results["test_precision"][-1],
        "recall": results["test_recall"][-1],
        "f1": results["test_f1"][-1]
    }
    
    return results, final_train_metrics, final_test_metrics

# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()
results, final_train_metrics, final_test_metrics = train(
    model=pretrained_vit,
    train_dataloader=train_dataloader_pretrained,
    test_dataloader=test_dataloader_pretrained,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler,
    epochs=15,
    device=device
)

# --- Cell ---

# Eğitim eğrilerini çiz
plot_training_curves(results)

# Test verisi için son değerlendirme
test_evaluation = evaluate_model(pretrained_vit, test_dataloader_pretrained, loss_fn, device)

# Konfüzyon matrisini çiz
plot_confusion_matrix(test_evaluation["y_true"], test_evaluation["y_pred"], class_names)

# Metrikleri tablo olarak göster
display_metrics_table(final_train_metrics, test_evaluation)

# Sınıflandırma raporu oluştur
classification_rep = classification_report(test_evaluation["y_true"], test_evaluation["y_pred"], 
                                          target_names=class_names)
print("Sınıflandırma Raporu:")
print(classification_rep)

# Modeli kaydet
torch.save(pretrained_vit.state_dict(), 'animal_classifier_vit.pth')
print("Model kaydedildi: animal_classifier_vit.pth")


