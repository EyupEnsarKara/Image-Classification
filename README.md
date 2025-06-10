# 🐾 Görüntü Sınıflandırma Projesi

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**Vision Transformer (ViT) kullanarak hayvan görüntülerini sınıflandıran yapay zeka projesi**

[🚀 Hızlı Başlangıç](#-hızlı-başlangıç) • [📊 Özellikler](#-özellikler) • [🛠️ Kurulum](#️-kurulum) • [📖 Kullanım](#-kullanım)

</div>

---

## 📋 İçindekiler

- [🎯 Proje Hakkında](#-proje-hakkında)
- [✨ Özellikler](#-özellikler)
- [🛠️ Kurulum](#️-kurulum)
- [🚀 Hızlı Başlangıç](#-hızlı-başlangıç)
- [📖 Kullanım](#-kullanım)
- [📁 Proje Yapısı](#-proje-yapısı)
- [📊 Çıktılar](#-çıktılar)
- [🔧 Yapılandırma](#-yapılandırma)
- [🤝 Katkıda Bulunma](#-katkıda-bulunma)

---

## 🎯 Proje Hakkında

Bu proje, **PyTorch** ve **Vision Transformer (ViT)** teknolojilerini kullanarak hayvan görüntülerini otomatik olarak sınıflandırmak için geliştirilmiş modern bir yapay zeka uygulamasıdır. Transfer learning yaklaşımı ile önceden eğitilmiş ViT-B/16 modelini kullanarak yüksek doğruluk oranları elde eder.

### 🎨 Neden Bu Proje?

- 🧠 **State-of-the-art Vision Transformer** teknolojisi
- 🚀 **Transfer Learning** ile hızlı ve etkili eğitim
- 📊 **Kapsamlı metrik analizi** ve görselleştirme
- 🔧 **Modüler yapı** ile kolay genişletilebilirlik

---

## ✨ Özellikler

### 🤖 Model Özellikleri
- **Vision Transformer (ViT-B/16)** mimarisi
- **Transfer Learning** ile önceden eğitilmiş ağırlıklar
- **Veri artırma** teknikleri ile robust eğitim
- **Learning Rate Scheduler** ile dinamik öğrenme oranı

### 📈 Analiz ve Görselleştirme
- **Kapsamlı metrik hesaplama** (Precision, Recall, F1-Score)
- **Confusion Matrix** görselleştirme
- **Eğitim eğrileri** ve kayıp fonksiyonu grafikleri
- **Otomatik test sonuçları** kaydetme

### 🛠️ Teknik Özellikler
- **Modüler kod yapısı** ile temiz organizasyon
- **JSON/CSV çıktı formatları** ile kolay entegrasyon
- **Seed kontrolü** ile tekrarlanabilir sonuçlar
- **Kapsamlı hata yönetimi**

---

## 🛠️ Kurulum

### Gereksinimler

- Python 3.8+
- CUDA destekli GPU (önerilen)

### 📦 Paket Kurulumu

```bash
# Repoyu klonlayın
git clone <repo-url>
cd Image-Classification

# Sanal ortam oluşturun (önerilen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

---

## 🚀 Hızlı Başlangıç

### 1️⃣ Veri Hazırlığı

Verilerinizi aşağıdaki yapıda organize edin:

```
yazlab-data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

### 2️⃣ Model Eğitimi

```bash
python train_model.py
```

### 3️⃣ Model Testi

```bash
python test_model.py
```

---

## 📖 Kullanım

### 🎯 Model Eğitimi

```python
from engine_functions import train_model
from utils import set_seeds

# Seed ayarlama
set_seeds(42)

# Model eğitimi başlatma
train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10
)
```

### 🔍 Tahmin Yapma

```python
from prediction_functions import pred_and_plot_image

# Tek görüntü üzerinde tahmin
pred_and_plot_image(
    model=model,
    image_path="path/to/image.jpg",
    class_names=class_names
)
```

### 📊 Sonuçları Görselleştirme

```python
from utils import plot_loss_curves

# Eğitim eğrilerini çizme
plot_loss_curves(results)
```

---

## 📁 Proje Yapısı

```
📦 Image-Classification/
├── 🎯 Ana Modüller
│   ├── train_model.py          # Model eğitimi
│   ├── test_model.py           # Model testi
│   ├── engine_functions.py     # Eğitim/test fonksiyonları
│   ├── prediction_functions.py # Tahmin fonksiyonları
│   └── utils.py               # Yardımcı fonksiyonlar
├── 📊 Çıktılar
│   ├── animal_classifier_vit.pth
│   ├── confusion_matrix.png
│   ├── model_metrics.png
│   └── test_results/
├── 📋 Yapılandırma
│   ├── requirements.txt
│   └── README.md
└── 📁 Veri
    └── yazlab-data/
```

### 🔧 Modül Detayları

#### `engine_functions.py`
```python
├── train_step()    # Tek epoch eğitimi
├── test_step()     # Tek epoch testi
└── train_model()   # Tam eğitim döngüsü
```

#### `prediction_functions.py`
```python
└── pred_and_plot_image()  # Tahmin + görselleştirme
```

#### `utils.py`
```python
├── plot_loss_curves()  # Eğitim eğrileri
├── set_seeds()         # Seed kontrolü
└── accuracy_fn()       # Doğruluk hesaplama
```

---

## 📊 Çıktılar

### 🎓 Eğitim Sonrası Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `animal_classifier_vit.pth` | 💾 Eğitilmiş model ağırlıkları |
| `confusion_matrix.png` | 📈 Confusion matrix görselleştirmesi |
| `model_metrics.png` | 📊 Detaylı metrik grafikleri |
| `model_metrics.json` | 📋 JSON formatında metrikler |

### 🧪 Test Sonrası Dosyalar

```
test_results/
├── 📊 all_results.json      # Tüm test sonuçları
├── 📋 summary.json          # Özet sonuçlar
├── 📈 results.csv           # CSV formatında sonuçlar
└── 📄 summary_report.txt    # Metin formatında rapor
```

---

## 🔧 Yapılandırma

### ⚙️ Model Parametreleri

```python
# Varsayılan ayarlar
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
IMAGE_SIZE = 224
```

### 📊 Metrik Ayarları

Proje aşağıdaki metrikleri otomatik olarak hesaplar:
- ✅ **Accuracy** (Doğruluk)
- 🎯 **Precision** (Kesinlik)
- 📈 **Recall** (Duyarlılık)
- ⚖️ **F1-Score** (F1 Skoru)

---

## 🤝 Katkıda Bulunma

Katkılarınızı memnuniyetle karşılarız! Lütfen aşağıdaki adımları takip edin:

1. 🍴 **Fork** edin
2. 🌟 **Feature branch** oluşturun (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** edin (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** edin (`git push origin feature/AmazingFeature`)
5. 🔄 **Pull Request** açın

---

## 📄 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır. Detaylar için `LICENSE` dosyasına bakın.

---

## 🙏 Teşekkürler

- **PyTorch** takımına harika framework için
- **Google Research** Vision Transformer modeli için
- **Hugging Face** önceden eğitilmiş modeller için

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!**

Made with ❤️ by [Your Name]

</div>
