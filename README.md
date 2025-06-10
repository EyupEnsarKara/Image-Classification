# Görüntü Sınıflandırma Projesi

Bu proje, PyTorch ve Vision Transformer (ViT) kullanarak hayvan görüntülerini sınıflandırmak için geliştirilmiştir.

## Proje Yapısı

### Ana Dosyalar

- **`train_model.py`**: Model eğitimi için ana script
- **`test_model.py`**: Eğitilmiş modeli test etmek için script
- **`engine_functions.py`**: Model eğitimi ve test fonksiyonları
- **`prediction_functions.py`**: Tahmin yapma fonksiyonları
- **`utils.py`**: Yardımcı fonksiyonlar (görselleştirme, seed ayarlama vb.)

### Fonksiyon Modülleri

#### `engine_functions.py`
- `train_step()`: Tek epoch için model eğitimi
- `test_step()`: Tek epoch için model testi
- `train_model()`: Tam eğitim döngüsü

#### `prediction_functions.py`
- `pred_and_plot_image()`: Tek görüntü üzerinde tahmin yapma ve görselleştirme

#### `utils.py`
- `plot_loss_curves()`: Eğitim eğrilerini çizme
- `set_seeds()`: Rastgele seed ayarlama
- `accuracy_fn()`: Doğruluk hesaplama

## Kullanım

### Model Eğitimi
```python
python train_model.py
```

### Model Testi
```python
python test_model.py
```

## Özellikler

- **Vision Transformer (ViT-B/16)** kullanımı
- **Transfer Learning** ile önceden eğitilmiş ağırlıklar
- **Veri artırma** teknikleri
- **Learning Rate Scheduler** ile dinamik öğrenme oranı
- **Kapsamlı metrik hesaplama** (Precision, Recall, F1-Score)
- **Confusion Matrix** görselleştirme
- **Otomatik test sonuçları** kaydetme

## Gereksinimler

Gerekli paketler `requirements.txt` dosyasında listelenmiştir:

```bash
pip install -r requirements.txt
```

## Veri Yapısı

Proje aşağıdaki veri yapısını bekler:

```
yazlab-data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Çıktılar

### Eğitim Sonrası
- `animal_classifier_vit.pth`: Eğitilmiş model ağırlıkları
- `confusion_matrix.png`: Confusion matrix görselleştirmesi
- `model_metrics.png`: Detaylı metrik görselleştirmeleri
- `model_metrics.json`: JSON formatında metrikler

### Test Sonrası
- `test_results/`: Test sonuçları klasörü
  - `all_results.json`: Tüm test sonuçları
  - `summary.json`: Özet sonuçlar
  - `results.csv`: CSV formatında sonuçlar
  - `summary_report.txt`: Metin formatında özet rapor

## Modüler Yapı

Proje, `going_modular` paketinden bağımsız hale getirilmiş ve tüm fonksiyonlar ana dizinde ayrı modüller halinde organize edilmiştir. Bu yapı:

- **Daha kolay bakım** sağlar
- **Fonksiyonları yeniden kullanılabilir** hale getirir
- **Kod organizasyonunu** iyileştirir
- **Import bağımlılıklarını** azaltır
