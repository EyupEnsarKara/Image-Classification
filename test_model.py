import torch
import torchvision
from prediction_functions import pred_and_plot_image
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from tqdm import tqdm
import pandas as pd
import json

# Cihazı ayarla
device = "cuda" if torch.cuda.is_available() else "cpu"

# Eğitilmiş modeli yükle
pretrained_vit = torchvision.models.vit_b_16()
# Sınıf isimlerini al
class_names = sorted(os.listdir('yazlab-data/val'))
# Model başlığını güncelle
pretrained_vit.heads = torch.nn.Linear(in_features=768, out_features=len(class_names))
# Model ağırlıklarını yükle
pretrained_vit.load_state_dict(torch.load('animal_classifier_vit.pth', map_location=device))
pretrained_vit.to(device)

def create_test_directory():
    """Test sonuçları için klasör oluşturur."""
    # Ana test klasörünü oluştur
    test_dir = "test_results"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Alt klasörleri oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_test_dir = os.path.join(test_dir, f"test_{timestamp}")
    os.makedirs(current_test_dir)
    
    return current_test_dir

def test_single_image(image_path, test_dir):
    """
    Tek bir görüntü üzerinde tahmin yapar ve sonucu kaydeder.
    
    Args:
        image_path (str): Test edilecek görüntünün yolu
        test_dir (str): Test sonuçlarının kaydedileceği klasör
    
    Returns:
        dict: Test sonuçları
    """
    try:
        # Görüntüyü yükle
        img = Image.open(image_path)
        
        # Görüntü dönüşümlerini uygula
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Görüntüyü dönüştür
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Modeli değerlendirme moduna al
        pretrained_vit.eval()
        
        # Tahmin yap
        with torch.no_grad():
            output = pretrained_vit(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
        # En yüksek olasılıklı sınıfı bul
        pred_prob, pred_class = torch.max(probabilities, 1)
        
        # Tüm sınıfların olasılıklarını al
        all_probs = probabilities[0].cpu().numpy()
        
        # Gerçek sınıfı al (klasör adından)
        true_class = os.path.basename(os.path.dirname(image_path))
        
        # Sonuçları hazırla
        result = {
            "image_path": image_path,
            "true_class": true_class,
            "predicted_class": class_names[pred_class.item()],
            "confidence": float(pred_prob.item() * 100),
            "is_correct": true_class == class_names[pred_class.item()],
            "class_probabilities": {class_name: float(prob * 100) for class_name, prob in zip(class_names, all_probs)}
        }
        
        # Görüntüyü kaydet
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(f"Gerçek: {true_class} | Tahmin: {result['predicted_class']} | Güven: {result['confidence']:.2f}%")
        plt.axis('off')
        
        # Görüntüyü kaydet
        image_name = os.path.basename(image_path)
        result_image_path = os.path.join(test_dir, f"pred_{image_name}")
        plt.savefig(result_image_path)
        plt.close()
        
        return result
        
    except Exception as e:
        return {
            "image_path": image_path,
            "error": str(e)
        }

def test_all_validation_images():
    """
    Tüm validation görüntülerini test eder ve sonuçları kaydeder.
    """
    # Test klasörünü oluştur
    test_dir = create_test_directory()
    
    # Validation klasöründeki tüm görüntüleri bul
    validation_dir = "yazlab-data/val"
    all_images = []
    for class_name in class_names:
        class_dir = os.path.join(validation_dir, class_name)
        if os.path.exists(class_dir):
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
            all_images.extend(images)
    
    print(f"\nToplam {len(all_images)} görüntü test edilecek...")
    
    # Sonuçları saklamak için liste
    results = []
    
    # Progress bar ile tüm görüntüleri test et
    for image_path in tqdm(all_images, desc="Görüntüler test ediliyor"):
        result = test_single_image(image_path, test_dir)
        results.append(result)
    
    # Sonuçları analiz et
    total_images = len(results)
    correct_predictions = sum(1 for r in results if r.get('is_correct', False))
    accuracy = (correct_predictions / total_images) * 100
    
    # Sınıf bazında doğruluk oranları
    class_accuracies = {}
    for class_name in class_names:
        class_images = [r for r in results if r.get('true_class') == class_name]
        if class_images:
            class_correct = sum(1 for r in class_images if r.get('is_correct', False))
            class_accuracies[class_name] = (class_correct / len(class_images)) * 100
    
    # Özet sonuçları hazırla
    summary = {
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": total_images,
        "correct_predictions": correct_predictions,
        "overall_accuracy": accuracy,
        "class_accuracies": class_accuracies
    }
    
    # Sonuçları kaydet
    # Tüm sonuçları JSON olarak kaydet
    with open(os.path.join(test_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Özet sonuçları JSON olarak kaydet
    with open(os.path.join(test_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    # Sonuçları CSV olarak kaydet
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(test_dir, "results.csv"), index=False)
    
    # Özet raporu oluştur
    report = f"""=== Test Sonuçları Özeti ===
Test Tarihi: {summary['test_date']}
Toplam Görüntü Sayısı: {summary['total_images']}
Doğru Tahmin Sayısı: {summary['correct_predictions']}
Genel Doğruluk Oranı: {summary['overall_accuracy']:.2f}%

Sınıf Bazında Doğruluk Oranları:
"""
    
    for class_name, acc in class_accuracies.items():
        report += f"{class_name}: {acc:.2f}%\n"
    
    # Özet raporu kaydet
    with open(os.path.join(test_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    
    # Sonuçları konsola yazdır
    print("\n" + report)
    print(f"\nDetaylı sonuçlar '{test_dir}' klasörüne kaydedildi:")
    print(f"- Tüm sonuçlar (JSON): {os.path.join(test_dir, 'all_results.json')}")
    print(f"- Özet sonuçlar (JSON): {os.path.join(test_dir, 'summary.json')}")
    print(f"- Sonuçlar (CSV): {os.path.join(test_dir, 'results.csv')}")
    print(f"- Özet rapor: {os.path.join(test_dir, 'summary_report.txt')}")

if __name__ == "__main__":
    test_all_validation_images() 