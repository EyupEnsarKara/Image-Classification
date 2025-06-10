"""
Tahmin yapmak için yardımcı fonksiyonlar.
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image

# Cihazı ayarla
device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    """Hedef görüntü üzerinde hedef model ile tahmin yapar.

    Args:
        model (torch.nn.Module): Görüntü üzerinde tahmin yapacak eğitilmiş PyTorch modeli.
        class_names (List[str]): Tahminleri eşlemek için hedef sınıfların listesi.
        image_path (str): Tahmin yapılacak hedef görüntünün dosya yolu.
        image_size (Tuple[int, int], optional): Hedef görüntünün dönüştürüleceği boyut. Varsayılan (224, 224).
        transform (torchvision.transforms, optional): Görüntü üzerinde uygulanacak dönüşüm. 
                                                     Varsayılan None, ImageNet normalizasyonu kullanır.
        device (torch.device, optional): Tahmin yapılacak hedef cihaz. Varsayılan device.
    """

    # Görüntüyü aç
    img = Image.open(image_path)

    # Görüntü için dönüşüm oluştur (eğer yoksa)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Görüntü üzerinde tahmin yap
    
    # Modelin hedef cihazda olduğundan emin ol
    model.to(device)

    # Model değerlendirme modunu ve çıkarım modunu aç
    model.eval()
    with torch.inference_mode():
        # Görüntüyü dönüştür ve ekstra boyut ekle
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Ekstra boyutlu görüntü üzerinde tahmin yap ve hedef cihaza gönder
        target_image_pred = model(transformed_image.to(device))

    # Logit'leri -> tahmin olasılıklarına dönüştür
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Tahmin olasılıklarını -> tahmin etiketlerine dönüştür
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Tahmin edilen etiket ve olasılık ile görüntüyü çiz
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Tahmin: {class_names[target_image_pred_label]} | Olasılık: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False) 