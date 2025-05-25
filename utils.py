"""
Kurs boyunca kullanılan yardımcı fonksiyonlar.

Bir kez tanımlanıp tekrar tekrar kullanılabilecek fonksiyonlar burada yer alır.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import os
import zipfile
from pathlib import Path
import requests
import os



# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
    Doğrusal eğitim ve test verilerini çizer ve tahminleri karşılaştırır.
    """
    plt.figure(figsize=(10, 7))

    # Eğitim verilerini mavi renkte çiz
    plt.scatter(train_data, train_labels, c="b", s=4, label="Eğitim verisi")

    # Test verilerini yeşil renkte çiz
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test verisi")

    if predictions is not None:
        # Tahminleri kırmızı renkte çiz
        plt.scatter(test_data, predictions, c="r", s=4, label="Tahminler")

    # Lejantı göster
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Gerçek etiketler ve tahminler arasındaki doğruluğu hesaplar.

    Args:
        y_true (torch.Tensor): Tahminler için gerçek etiketler.
        y_pred (torch.Tensor): Karşılaştırılacak tahminler.

    Returns:
        [torch.float]: y_true ve y_pred arasındaki doğruluk değeri, örn. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Başlangıç ve bitiş zamanı arasındaki farkı yazdırır.

    Args:
        start (float): Hesaplamanın başlangıç zamanı (timeit formatında tercih edilir). 
        end (float): Hesaplamanın bitiş zamanı.
        device ([type], optional): Hesaplamanın çalıştığı cihaz. Varsayılan None.

    Returns:
        float: başlangıç ve bitiş arasındaki saniye cinsinden süre (yüksek değer daha uzun süre).
    """
    total_time = end - start
    print(f"\n{device} üzerinde eğitim süresi: {total_time:.3f} saniye")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Sonuç sözlüğünün eğitim eğrilerini çizer.

    Args:
        results (dict): değer listelerini içeren sözlük, örn.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Kayıp grafiğini çiz
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="eğitim_kaybı")
    plt.plot(epochs, test_loss, label="test_kaybı")
    plt.title("Kayıp")
    plt.xlabel("Epoch'lar")
    plt.legend()

    # Doğruluk grafiğini çiz
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="eğitim_doğruluğu")
    plt.plot(epochs, test_accuracy, label="test_doğruluğu")
    plt.title("Doğruluk")
    plt.xlabel("Epoch'lar")
    plt.legend()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Eğitilmiş model ile hedef görüntü üzerinde tahmin yapar ve görüntüyü çizer.

    Args:
        model (torch.nn.Module): eğitilmiş PyTorch görüntü sınıflandırma modeli.
        image_path (str): hedef görüntünün dosya yolu.
        class_names (List[str], optional): hedef görüntü için farklı sınıf isimleri. Varsayılan None.
        transform (_type_, optional): hedef görüntünün dönüşümü. Varsayılan None.
        device (torch.device, optional): hesaplama yapılacak hedef cihaz. Varsayılan "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Hedef görüntünün ve model tahmininin başlık olarak gösterildiği Matplotlib grafiği.

    Örnek kullanım:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["sınıf_1", "sınıf_2", "sınıf_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # Görüntüyü yükle ve tensor değerlerini float32'ye dönüştür
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Görüntü piksel değerlerini 255'e bölerek [0, 1] aralığına getir
    target_image = target_image / 255.0

    # Gerekirse dönüştür
    if transform:
        target_image = transform(target_image)

    # Modelin hedef cihazda olduğundan emin ol
    model.to(device)

    # Model değerlendirme modunu ve çıkarım modunu aç
    model.eval()
    with torch.inference_mode():
        # Görüntüye ekstra boyut ekle
        target_image = target_image.unsqueeze(dim=0)

        # Ekstra boyutlu görüntü üzerinde tahmin yap ve hedef cihaza gönder
        target_image_pred = model(target_image.to(device))

    # Logit'leri -> tahmin olasılıklarına dönüştür (çok sınıflı sınıflandırma için torch.softmax() kullan)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Tahmin olasılıklarını -> tahmin etiketlerine dönüştür
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Görüntüyü tahmin ve tahmin olasılığı ile birlikte çiz
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )
    if class_names:
        title = f"Tahmin: {class_names[target_image_pred_label.cpu()]} | Olasılık: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Tahmin: {target_image_pred_label} | Olasılık: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Torch işlemleri için rastgele tohum ayarlar.

    Args:
        seed (int, optional): Ayarlanacak rastgele tohum. Varsayılan 42.
    """
    # Genel torch işlemleri için tohum ayarla
    torch.manual_seed(seed)
    # CUDA torch işlemleri için tohum ayarla (GPU'da gerçekleşenler)
    torch.cuda.manual_seed(seed) 