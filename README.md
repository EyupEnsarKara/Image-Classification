# 🐾 Hayvan Görüntü Sınıflandırma Projesi

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![Vision Transformer](https://img.shields.io/badge/Model-Vision%20Transformer-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**Vision Transformer (ViT) kullanarak 90+ hayvan türünü sınıflandıran gelişmiş yapay zeka projesi**

[🚀 Hızlı Başlangıç](#-hızlı-başlangıç) • [🎮 GUI Kullanımı](#-gui-kullanımı) • [📊 Özellikler](#-özellikler) • [🛠️ Kurulum](#️-kurulum) • [📖 Kullanım](#-kullanım)

---

### 🌟 **Demo Görselleri**

| 🎯 **Eğitim Süreçi** | 🖥️ **GUI Arayüzü** | 📊 **Analiz Sonuçları** |
|:---:|:---:|:---:|
| ![Eğitim](https://via.placeholder.com/200x150/2b2b2b/ffffff?text=Eğitim+Süreçi) | ![GUI](https://via.placeholder.com/200x150/0078d4/ffffff?text=Modern+GUI) | ![Analiz](https://via.placeholder.com/200x150/107c10/ffffff?text=Detaylı+Analiz) |

</div>

---

## 📋 İçindekiler

- [🎯 Proje Hakkında](#-proje-hakkında)
- [✨ Özellikler](#-özellikler)
- [🛠️ Kurulum](#️-kurulum)
- [🚀 Hızlı Başlangıç](#-hızlı-başlangıç)
- [🎮 GUI Kullanımı](#-gui-kullanımı)
- [📖 Komut Satırı Kullanımı](#-komut-satırı-kullanımı)
- [📁 Proje Yapısı](#-proje-yapısı)
- [🧠 Model Detayları](#-model-detayları)
- [📊 Çıktılar ve Sonuçlar](#-çıktılar-ve-sonuçlar)
- [🔧 Yapılandırma](#-yapılandırma)
- [❓ Sorun Giderme](#-sorun-giderme)
- [🤝 Katkıda Bulunma](#-katkıda-bulunma)

---

## 🎯 Proje Hakkında

Bu proje, **state-of-the-art Vision Transformer (ViT)** teknolojisini kullanarak **90+ farklı hayvan türünü** otomatik olarak sınıflandıran kapsamlı bir yapay zeka uygulamasıdır. Transfer learning yaklaşımı ile önceden eğitilmiş ViT-B/16 modelini kullanarak yüksek doğruluk oranları elde eder.

### 🎨 **Ne Yapabilir?**

- 🔍 **Görüntü Analizi**: Tek bir fotoğraftan hayvan türünü tahmin etme
- 📱 **Kullanıcı Dostu Arayüz**: Modern GUI ile kolay kullanım
- 📊 **Toplu Test**: Birden fazla görüntüyü aynı anda analiz etme
- 📈 **Detaylı Raporlama**: Kapsamlı analiz ve güven skorları
- 🎯 **Yüksek Doğruluk**: Transfer learning ile optimize edilmiş performans

### 🦎 **Desteklenen Hayvan Türleri (90+ Sınıf)**

<details>
<summary>📖 <strong>Tüm Desteklenen Hayvanları Görüntüle</strong></summary>

**🐾 Memeli Hayvanlar (50):**
antelope, badger, bat, bear, bison, boar, cat, chimpanzee, cow, coyote, deer, dog, dolphin, donkey, elephant, fox, goat, gorilla, hamster, hare, hedgehog, hippopotamus, horse, hyena, kangaroo, koala, leopard, lion, lizard, mouse, okapi, orangutan, otter, ox, panda, pig, porcupine, possum, raccoon, rat, reindeer, rhinoceros, seal, sheep, squirrel, tiger, whale, wolf, wombat, zebra

**🐦 Kuşlar (17):**
crow, duck, eagle, flamingo, goose, hornbill, hummingbird, owl, parrot, pelecaniformes, penguin, pigeon, sandpiper, sparrow, swan, turkey, woodpecker

**🐛 Böcekler (11):**
bee, beetle, butterfly, caterpillar, cockroach, dragonfly, fly, grasshopper, ladybugs, mosquito, moth

**🐠 Deniz Canlıları (9):**
goldfish, jellyfish, lobster, octopus, oyster, seahorse, shark, squid, starfish

**🐢 Sürüngenler ve Diğerleri (3):**
snake, turtle, crab

</details>

---

## ✨ Özellikler

### 🤖 **Model ve AI Özellikleri**
| Özellik | Açıklama |
|---------|----------|
| 🧠 **Vision Transformer (ViT-B/16)** | Google Research tarafından geliştirilen state-of-the-art model |
| 🚀 **Transfer Learning** | Önceden eğitilmiş ağırlıklarla hızlı ve etkili eğitim |
| 📊 **90+ Sınıf Desteği** | Geniş hayvan türü yelpazesi |
| 🎯 **Yüksek Doğruluk** | Optimize edilmiş eğitim süreci ile yüksek performans |
| 🔄 **Veri Artırma** | Robust model için gelişmiş veri augmentasyon teknikleri |

### 🖥️ **Kullanıcı Arayüzü Özellikleri**
| Özellik | Açıklama |
|---------|----------|
| 🎨 **Modern GUI** | CustomTkinter ile geliştirilmiş kullanıcı dostu arayüz |
| 📱 **Responsive Tasarım** | Farklı ekran boyutlarına uyumlu |
| 🌙 **Dark Theme** | Göz yormayan modern karanlık tema |
| 📊 **Gerçek Zamanlı Analiz** | Anlık tahmin sonuçları ve güven skorları |
| 🔄 **Toplu İşlem** | Birden fazla görüntüyü aynı anda test etme |
| 📦 **ZIP Desteği** | Sıkıştırılmış dosyalardan direkt analiz |

### 📈 **Analiz ve Raporlama**
| Özellik | Açıklama |
|---------|----------|
| 📊 **Detaylı Metrikler** | Precision, Recall, F1-Score hesaplama |
| 🎯 **Confusion Matrix** | Görsel sınıflandırma matrisi |
| 📋 **Kapsamlı Raporlar** | JSON, CSV, TXT formatlarında çıktı |
| 📈 **Eğitim Eğrileri** | Kayıp ve doğruluk grafikleri |
| 🔍 **Güven Skoru Analizi** | Her tahmin için detaylı güvenilirlik bilgisi |

### ⚡ **Teknik Özellikler**
| Özellik | Açıklama |
|---------|----------|
| 🔧 **Modüler Yapı** | Temiz ve genişletilebilir kod organizasyonu |
| 🎛️ **Otomatik Cihaz Algılama** | GPU/CPU otomatik seçimi |
| 📱 **Cross-Platform** | Windows, macOS, Linux desteği |
| 🔄 **Otomatik Model Uyumluluk** | Farklı model yapılarını otomatik algılama |
| 💾 **Düşük Bellek Kullanımı** | Optimize edilmiş bellek yönetimi |

---

## 🛠️ Kurulum

### 📋 **Sistem Gereksinimleri**

| Gereksinim | Minimum | Önerilen |
|------------|---------|----------|
| 🐍 **Python** | 3.8+ | 3.9+ |
| 💾 **RAM** | 4 GB | 8 GB+ |
| 💿 **Disk Alanı** | 2 GB | 5 GB+ |
| 🎮 **GPU** | Opsiyonel | CUDA destekli |

### 📦 **Adım Adım Kurulum**

#### 1️⃣ **Projeyi İndirin**
```bash
git clone <repository-url>
cd Image-Classification
```

#### 2️⃣ **Sanal Ortam Oluşturun** (Önerilen)
```bash
# Python sanal ortamı oluşturun
python -m venv ai_env

# Sanal ortamı aktifleştirin
# Windows:
ai_env\Scripts\activate
# macOS/Linux:
source ai_env/bin/activate
```

#### 3️⃣ **Gerekli Paketleri Yükleyin**
```bash
pip install -r requirements.txt
```

#### 4️⃣ **Veri Klasörünü Hazırlayın**
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

---

## 🚀 Hızlı Başlangıç

### ⚡ **30 Saniyede Başlayın!**

```bash
# 1. GUI arayüzünü başlatın
python simple_model_gui.py

# 2. Bir görüntü seçin
# 3. "Tahmin Yap" butonuna tıklayın
# 4. Sonuçları görün!
```

### 📋 **Temel Kullanım Senaryoları**

#### 🎯 **Senaryo 1: Tek Görüntü Analizi**
```bash
# GUI ile tek görüntü analizi
python simple_model_gui.py
```

#### 📊 **Senaryo 2: Model Eğitimi**
```bash
# Yeni model eğitimi
python train_model.py
```

#### 🧪 **Senaryo 3: Kapsamlı Test**
```bash
# Tüm validation setini test etme
python test_model.py
```

---

## 🎮 GUI Kullanımı

### 🖥️ **Modern Arayüz Özellikleri**

Projemiz **iki farklı GUI seçeneği** sunar:

#### 🎨 **Seçenek 1: Basit GUI (Önerilen)**
```bash
python simple_model_gui.py
```

**✨ Özellikler:**
- 🚀 Hızlı başlatma
- 💻 Standart tkinter kullanır
- 🎯 Kullanımı kolay
- 🔧 Düşük sistem gereksinimi

#### 🌟 **Seçenek 2: Gelişmiş GUI**
```bash
python model_test_gui.py  # (CustomTkinter gerektirir)
```

**✨ Özellikler:**
- 🎨 Modern dark theme
- 🔄 Smooth animasyonlar
- 📱 Responsive tasarım

### 📖 **GUI Kullanım Rehberi**

#### **1️⃣ Arayüzü Başlatın**
```bash
python simple_model_gui.py
```

#### **2️⃣ Model Durumunu Kontrol Edin**
- ✅ **Yeşil işaret**: Model hazır
- ⚠️ **Sarı işaret**: Yükleniyor
- ❌ **Kırmızı işaret**: Hata var

#### **3️⃣ Görüntü Seçin**
| Yöntem | Açıklama |
|--------|----------|
| 📁 **Tek Dosya** | "Görüntü Seç" ile tek dosya seçimi |
| 📦 **ZIP Dosyası** | "ZIP Seç" ile toplu görüntü yükleme |
| 🖱️ **Sürükle-Bırak** | Dosyaları direkt sürükleyip bırakma |

#### **4️⃣ Analiz Yapın**
```
🔮 Tahmin Yap → 📊 Sonuçları Görün → 💾 Kaydet
```

#### **5️⃣ Sonuçları Yorumlayın**

**📊 Güven Skoru Rehberi:**
| Skor | Durum | Açıklama |
|------|-------|----------|
| 🟢 **90-100%** | Mükemmel | Çok yüksek güven |
| 🟡 **70-89%** | İyi | Yüksek güven |
| 🟠 **50-69%** | Orta | Orta güven |
| 🔴 **0-49%** | Düşük | Tekrar değerlendirin |

### 🎯 **Gelişmiş GUI Özellikleri**

#### **📊 Toplu Test Özelliği**
- ✅ Birden fazla görüntüyü aynı anda analiz
- 📈 Batch sonuçlarının karşılaştırılması
- 📋 Otomatik rapor oluşturma
- 💾 Sonuçları farklı formatlarda kaydetme

#### **📦 ZIP Dosya Desteği**
- ✅ Sıkıştırılmış dosyalardan direkt okuma
- 🔍 ZIP içeriğini göz atma
- ⚡ Hızlı toplu işlem
- 💾 Bellek dostu çözüm

---

## 📖 Komut Satırı Kullanımı

### 🎯 **Model Eğitimi**

```bash
python train_model.py
```

**⚙️ Eğitim Parametreleri:**
```python
EPOCHS = 15                    # Eğitim döngü sayısı
BATCH_SIZE = 8                # Batch boyutu  
LEARNING_RATE = 1e-4          # Öğrenme oranı
IMAGE_SIZE = 224              # Görüntü boyutu
```

**📊 Eğitim Süreci:**
1. 🔄 **Veri Yükleme**: Training ve validation setlerini hazırlama
2. 🧠 **Model Hazırlama**: ViT modelini transfer learning ile ayarlama
3. 🎯 **Eğitim**: 15 epoch boyunca model eğitimi
4. 📈 **Değerlendirme**: Her epoch'ta test performansı
5. 💾 **Kaydetme**: En iyi modeli otomatik kaydetme

### 🧪 **Model Testi**

```bash
python test_model.py
```

**📋 Test Özellikleri:**
- ✅ Tüm validation setini otomatik test
- 📊 Sınıf bazında doğruluk oranları
- 🎯 Confusion matrix oluşturma
- 📄 Detaylı raporlar (JSON, CSV, TXT)
- 🖼️ Tahmin görüntülerini kaydetme

### 🔍 **Model Kontrolü**

```bash
python check_model.py
```

**📊 Kontrol Edilen Özellikler:**
- 🏗️ Model yapısı uyumluluğu
- 🔢 Sınıf sayısı kontrolü
- ⚙️ Katman yapısı analizi
- 🎯 Olası sorunların tespiti

---

## 📁 Proje Yapısı

```
📦 Image-Classification/
├── 🎯 Ana Uygulamalar
│   ├── 🖥️ simple_model_gui.py      # Basit GUI arayüzü (BAŞLANGIC)
│   ├── 🌟 model_test_gui.py        # Gelişmiş GUI (CustomTkinter)
│   ├── 🏋️ train_model.py           # Model eğitimi
│   ├── 🧪 test_model.py            # Model testi
│   └── 🔍 check_model.py           # Model kontrolü
│
├── 🔧 Çekirdek Modüller
│   ├── ⚙️ engine_functions.py      # Eğitim/test fonksiyonları
│   ├── 🔮 prediction_functions.py  # Tahmin fonksiyonları
│   └── 🛠️ utils.py                 # Yardımcı fonksiyonlar
│
├── 📊 Veri ve Modeller
│   ├── 💾 animal_classifier_vit.pth # Eğitilmiş model
│   ├── 📁 yazlab-data/             # Eğitim verisi
│   └── 📋 test_results/            # Test sonuçları
│
├── 📋 Dokümantasyon
│   ├── 📖 README.md               # Ana rehber (BU DOSYA)
│   ├── 🎮 GUI_README.md           # GUI kullanım rehberi
│   └── 📊 rapor.tex               # Teknik rapor
│
└── ⚙️ Yapılandırma
    ├── 📦 requirements.txt         # Python paketleri
    └── 🔧 .gitattributes          # Git yapılandırması
```

### 🔧 **Modül Detayları**

#### **`engine_functions.py`** - Eğitim Motoru
```python
📊 train_step()      # Tek epoch eğitimi
🧪 test_step()       # Tek epoch testi  
🏋️ train_model()     # Tam eğitim döngüsü
```

#### **`prediction_functions.py`** - Tahmin Sistemi
```python
🔮 pred_and_plot_image()  # Tahmin + görselleştirme
📊 batch_predict()        # Toplu tahmin
```

#### **`utils.py`** - Yardımcı Araçlar
```python
📈 plot_loss_curves()    # Eğitim eğrileri
🎲 set_seeds()           # Seed kontrolü  
🎯 accuracy_fn()         # Doğruluk hesaplama
```

---

## 🧠 Model Detayları

### 🏗️ **Vision Transformer (ViT) Mimarisi**

| Özellik | Değer |
|---------|-------|
| 🔧 **Model Tipi** | Vision Transformer B/16 |
| 📐 **Input Boyutu** | 224 x 224 x 3 |
| 🧠 **Parametre Sayısı** | ~86M parameters |
| 🎯 **Sınıf Sayısı** | 90+ hayvan türü |
| ⚡ **Inference Hızı** | ~50ms (GPU) |

### 🚀 **Transfer Learning Yaklaşımı**

#### **📋 Eğitim Stratejisi**
1. 🔄 **Önceden Eğitilmiş Model**: ImageNet ağırlıkları
2. 🎯 **Fine-tuning**: Tüm katmanları eğitilebilir hale getirme
3. 🔧 **Sınıflandırıcı Değişimi**: Son katmanı 90 sınıf için ayarlama
4. 📊 **Veri Artırma**: Robust eğitim için augmentasyon

#### **🔧 Eğitim Parametreleri**
```python
# Optimizer ayarları
optimizer = torch.optim.Adam(lr=1e-4)
scheduler = ReduceLROnPlateau(patience=3)

# Veri artırma teknikleri
transforms = [
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.2),
    RandomResizedCrop(224, scale=(0.8, 1.0)),
    RandomGrayscale(p=0.1),
    RandomErasing(p=0.1)
]
```

### 📊 **Model Performansı**

#### **🎯 Beklenen Doğruluk Oranları**
| Kategori | Doğruluk |
|----------|----------|
| 🐾 **Genel** | %85-95 |
| 🐕 **Evcil Hayvanlar** | %90-98 |
| 🦁 **Büyük Memeliler** | %88-95 |
| 🐦 **Kuşlar** | %80-90 |
| 🐛 **Böcekler** | %75-85 |

---

## 📊 Çıktılar ve Sonuçlar

### 🏋️ **Eğitim Sonrası Dosyalar**

| Dosya | Açıklama | Boyut |
|-------|----------|-------|
| 💾 `animal_classifier_vit.pth` | Eğitilmiş model ağırlıkları | ~330MB |
| 📈 `confusion_matrix.png` | Confusion matrix görselleştirmesi | ~200KB |
| 📊 `model_metrics.png` | Detaylı metrik grafikleri | ~150KB |
| 📋 `model_metrics.json` | JSON formatında metrikler | ~5KB |
| 📈 `training_curves.png` | Eğitim loss/accuracy eğrileri | ~100KB |

### 🧪 **Test Sonrası Dosyalar**

```
📁 test_results/
├── 📊 all_results.json          # Tüm test sonuçları (detaylı)
├── 📋 summary.json              # Özet sonuçlar
├── 📈 results.csv               # Excel'de açılabilir sonuçlar
├── 📄 summary_report.txt        # İnsan okunabilir rapor
├── 🖼️ prediction_samples/       # Örnek tahmin görüntüleri
│   ├── correct_predictions/     # Doğru tahminler
│   └── wrong_predictions/       # Yanlış tahminler
└── 📊 class_analysis.json       # Sınıf bazında detaylı analiz
```

### 📋 **Çıktı Formatları**

#### **📊 JSON Çıktısı Örneği**
```json
{
  "test_date": "2024-01-15 14:30:25",
  "total_images": 1000,
  "correct_predictions": 875,
  "overall_accuracy": 87.5,
  "class_accuracies": {
    "dog": 95.2,
    "cat": 92.8,
    "lion": 89.5,
    "elephant": 91.0
  },
  "confusion_matrix": "[[...]]",
  "classification_report": "..."
}
```

#### **📈 CSV Çıktısı Örneği**
```csv
image_path,true_class,predicted_class,confidence,is_correct
val/dog/img1.jpg,dog,dog,0.95,True
val/cat/img2.jpg,cat,cat,0.88,True
val/lion/img3.jpg,lion,tiger,0.65,False
```

---

## 🔧 Yapılandırma

### ⚙️ **Model Hiperparametreleri**

```python
# 🏋️ Eğitim Ayarları
EPOCHS = 15                    # Eğitim döngü sayısı
BATCH_SIZE = 8                # Bellek kullanımına göre ayarlayın
LEARNING_RATE = 1e-4          # Adam optimizer için
IMAGE_SIZE = 224              # ViT standart input boyutu

# 📊 Veri Ayarları  
TRAIN_RATIO = 0.8             # Eğitim verisi oranı
VAL_RATIO = 0.2               # Validation verisi oranı
NUM_WORKERS = 4               # DataLoader worker sayısı

# 🎯 Model Ayarları
MODEL_NAME = "vit_b_16"       # Vision Transformer variant
PRETRAINED = True             # Transfer learning kullanımı
FREEZE_LAYERS = False         # Tüm katmanları eğitilebilir yap
```

### 🎨 **GUI Özelleştirme**

#### **🎨 Renk Teması Değiştirme**
```python
# simple_model_gui.py içinde
self.colors = {
    'bg': '#2b2b2b',          # Ana arka plan
    'fg': '#ffffff',          # Metin rengi
    'accent': '#0078d4',      # Vurgu rengi (mavi)
    'success': '#107c10',     # Başarı rengi (yeşil)
    'warning': '#ff8c00',     # Uyarı rengi (turuncu)
    'error': '#d13438',       # Hata rengi (kırmızı)
    'card': '#3c3c3c'         # Kart arka planı
}
```

#### **📐 Pencere Boyutu Ayarlama**
```python
# Pencere boyutunu değiştirin
self.root.geometry("1600x1000")  # Genişlik x Yükseklik
```

### 🔧 **Performans Optimizasyonu**

#### **💾 Bellek Kullanımı**
```python
# Düşük bellek için batch size'ı küçültün
BATCH_SIZE = 4      # 4GB RAM için
BATCH_SIZE = 8      # 8GB RAM için  
BATCH_SIZE = 16     # 16GB+ RAM için
```

#### **⚡ GPU Optimizasyonu**
```python
# GPU bellek temizleme
torch.cuda.empty_cache()

# Mixed precision training (isteğe bağlı)
from torch.cuda.amp import autocast, GradScaler
```

---

## ❓ Sorun Giderme

### 🚨 **Yaygın Hatalar ve Çözümleri**

#### **❌ Model Yükleme Hataları**

**Problem:** `FileNotFoundError: animal_classifier_vit.pth not found`
```bash
# Çözüm: Model dosyasını kontrol edin
ls -la *.pth  # Model dosyası var mı?
python check_model.py  # Model yapısını kontrol edin
```

**Problem:** `RuntimeError: size mismatch for heads.weight`
```python
# Çözüm: Model dosyası otomatik olarak uyarlanır
# Konsol çıktısını kontrol edin, uyarı mesajları normal
```

#### **💾 Bellek Hataları**

**Problem:** `CUDA out of memory`
```python
# Çözüm 1: Batch size'ı küçültün
BATCH_SIZE = 4  # train_model.py içinde

# Çözüm 2: GPU belleğini temizleyin  
torch.cuda.empty_cache()

# Çözüm 3: CPU kullanın
device = "cpu"  # GPU yerine CPU kullan
```

#### **🖼️ Görüntü Yükleme Hataları**

**Problem:** `PIL cannot identify image file`
```bash
# Çözüm: Desteklenen formatları kullanın
# Desteklenen: .jpg, .jpeg, .png, .bmp, .gif
# Desteklenmeyen: .webp, .tiff, .raw
```

**Problem:** `Turkish character error in file path`
```bash
# Çözüm: Dosya yolunda Türkçe karakter kullanmayın
# Yanlış: C:/Users/Öğrenci/görüntü.jpg
# Doğru: C:/Users/Student/image.jpg
```

#### **🔧 GUI Hataları**

**Problem:** `ModuleNotFoundError: No module named 'customtkinter'`
```bash
# Çözüm: Basit GUI kullanın
python simple_model_gui.py  # CustomTkinter gerektirmez

# Veya CustomTkinter yükleyin  
pip install customtkinter
```

### 🔍 **Performans Sorunları**

#### **🐌 Yavaş Tahmin**
```python
# Çözüm 1: Görüntü boyutunu kontrol edin
max_size = 224  # Çok büyük görüntüleri yeniden boyutlandırın

# Çözüm 2: Batch prediction kullanın
# Tek tek yerine toplu tahmin yapın

# Çözüm 3: Model precision'ı azaltın
model.half()  # Float16 kullanın (GPU'da)
```

#### **📊 Düşük Doğruluk**
```bash
# Çözüm 1: Daha fazla epoch ile eğitin
EPOCHS = 25  # 15 yerine 25 epoch

# Çözüm 2: Learning rate'i ayarlayın
LEARNING_RATE = 5e-5  # Daha küçük öğrenme oranı

# Çözüm 3: Veri kalitesini kontrol edin
# Etiketler doğru mu? Görüntüler net mi?
```

### 🛠️ **Debug Modunu Aktifleştirin**

```python
# train_model.py başına ekleyin
import logging
logging.basicConfig(level=logging.DEBUG)

# Detaylı hata mesajları için
import traceback
try:
    # Problemli kod
    pass
except Exception as e:
    traceback.print_exc()
```

### 📞 **Yardım Alma**

1. 📋 **Log dosyalarını kontrol edin**
2. 🔍 **Error mesajının tamamını kopyalayın**
3. 💻 **Sistem bilgilerinizi paylaşın** (OS, Python version, GPU)
4. 📊 **Model ve veri yapısını kontrol edin**

---

## 🤝 Katkıda Bulunma

### 🚀 **Nasıl Katkıda Bulunabilirsiniz?**

#### **🐛 Bug Raporlama**
1. 🔍 Issue tracker'da benzer sorun var mı kontrol edin
2. 📝 Detaylı bug raporu oluşturun
3. 🖼️ Ekran görüntüleri ekleyin
4. 💻 Sistem bilgilerini paylaşın

#### **✨ Yeni Özellik Önerisi**
1. 💡 Özellik önerinizi açıklayın
2. 🎯 Kullanım senaryosunu belirtin
3. 🔧 Mümkünse teknik detayları ekleyin

#### **🔧 Kod Katkısı**
```bash
# 1. Repo'yu fork edin
git clone https://github.com/yourusername/Image-Classification.git

# 2. Feature branch oluşturun  
git checkout -b feature/amazing-feature

# 3. Değişikliklerinizi commit edin
git commit -m "feat: Add amazing feature"

# 4. Branch'i push edin
git push origin feature/amazing-feature

# 5. Pull Request oluşturun
```

### 📋 **Katkı Rehberi**

#### **📝 Kod Standartları**
```python
# PEP 8 standardlarını takip edin
# Fonksiyon dokümantasyonu ekleyin
def example_function(param1: str) -> bool:
    """
    Fonksiyon açıklaması.
    
    Args:
        param1: Parametre açıklaması
        
    Returns:
        Dönüş değeri açıklaması
    """
    pass
```

#### **🧪 Test Etme**
```bash
# Değişikliklerinizi test edin
python train_model.py    # Eğitim çalışıyor mu?
python test_model.py     # Test çalışıyor mu?
python simple_model_gui.py  # GUI çalışıyor mu?
```

#### **🎯 İyileştirme Alanları**

| Alan | Öncelik | Açıklama |
|------|---------|----------|
| 🚀 **Performans** | Yüksek | Model inference hızını artırma |
| 🎨 **UI/UX** | Orta | Arayüz iyileştirmeleri |
| 📊 **Analitik** | Orta | Daha detaylı raporlama |
| 🔧 **Kod Kalitesi** | Yüksek | Refactoring ve optimize etme |
| 📱 **Cross-platform** | Düşük | Mac/Linux uyumluluğu |

---

## 📄 Lisans

Bu proje **MIT Lisansı** altında dağıtılmaktadır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

### 📋 **Lisans Özeti**
- ✅ **Ticari kullanım** izinli
- ✅ **Değiştirme** izinli  
- ✅ **Dağıtım** izinli
- ✅ **Özel kullanım** izinli
- ❗ **Sorumluluk** yok
- ❗ **Garanti** yok

---

## 🙏 Teşekkürler

### 🏆 **Kullanılan Teknolojiler**
- 🔥 **[PyTorch](https://pytorch.org/)** - Deep learning framework
- 🤖 **[Google Research](https://github.com/google-research/vision_transformer)** - Vision Transformer modeli
- 🤗 **[Hugging Face](https://huggingface.co/)** - Önceden eğitilmiş modeller
- 🎨 **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)** - Modern GUI framework
- 📊 **[Matplotlib](https://matplotlib.org/)** - Görselleştirme kütüphanesi

### 🎓 **Eğitim Kaynakları**
- 📚 **[PyTorch Tutorials](https://pytorch.org/tutorials/)**
- 🎥 **[Zero to Mastery PyTorch Course](https://www.learnpytorch.io/)**
- 📖 **[Papers With Code](https://paperswithcode.com/)**

### 👥 **Topluluk**
- 💬 **[PyTorch Discussions](https://discuss.pytorch.org/)**
- 🐦 **[AI Twitter Community](https://twitter.com/hashtag/PyTorch)**
- 🤝 **[Reddit r/MachineLearning](https://reddit.com/r/MachineLearning)**

---

<div align="center">

## 🌟 **Projeyi Beğendiyseniz Yıldız Vermeyi Unutmayın!** ⭐

[![Star on GitHub](https://img.shields.io/github/stars/username/Image-Classification.svg?style=social)](https://github.com/username/Image-Classification/stargazers)

**Made with ❤️ and 🤖 AI**

---

### 📈 **Proje İstatistikleri**

![GitHub repo size](https://img.shields.io/github/repo-size/username/Image-Classification)
![GitHub code size](https://img.shields.io/github/languages/code-size/username/Image-Classification)
![GitHub last commit](https://img.shields.io/github/last-commit/username/Image-Classification)
![GitHub issues](https://img.shields.io/github/issues/username/Image-Classification)

---

**🚀 Happy Coding! | 🤖 AI ile Geleceği Keşfedin!**

</div>
