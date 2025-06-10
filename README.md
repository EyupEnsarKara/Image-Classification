# 🔍 Vision Transformer ile Görüntü Sınıflandırma Projesi

Bu proje, **Vision Transformer (ViT)** teknolojisini kullanarak görüntü sınıflandırma işlemleri gerçekleştiren kapsamlı bir Python uygulamasıdır. Proje hem sıfırdan ViT modeli oluşturma hem de önceden eğitilmiş model kullanma yaklaşımlarını içermektedir.

## 🌟 Özellikler

- **🔧 Sıfırdan ViT Modeli**: Vision Transformer mimarisini sıfırdan inşa eden kod
- **🚀 Önceden Eğitilmiş Model**: PyTorch'un ViT-B-16 modelini kullanan transfer learning yaklaşımı
- **📊 Kapsamlı Test Suite**: Tüm validation veri seti üzerinde detaylı performans analizi
- **📈 Görselleştirme**: Eğitim grafikleri ve tahmin sonuçlarının görsel analizi
- **🔍 Modüler Tasarım**: Yeniden kullanılabilir kod yapısı
- **📋 Detaylı Raporlama**: JSON, CSV ve metin formatlarında sonuç raporları

## 🏗️ Proje Yapısı

```
Image-Classification/
├── 📁 going_modular/           # Modüler kod yapısı
│   └── going_modular/
│       ├── engine.py           # Eğitim döngüsü
│       ├── model_builder.py    # Model oluşturma fonksiyonları
│       ├── predictions.py      # Tahmin fonksiyonları
│       ├── train.py           # Eğitim scripti
│       └── utils.py           # Yardımcı fonksiyonlar
├── 📄 image_classifier_from_scratch.py    # Sıfırdan ViT implementasyonu
├── 📄 train_using_pretrained_model_image_classifier.py  # Transfer learning
├── 📄 test_model.py           # Model test ve değerlendirme
├── 📄 helper_functions.py     # Yardımcı fonksiyonlar
└── 📄 README.md              # Bu dosya
```

## 🧠 Vision Transformer Mimarisi

Bu projede implementasyonu yapılan Vision Transformer aşağıdaki bileşenleri içerir:

### 🔹 Ana Bileşenler
- **Patch Embedding**: Görüntüleri sabit boyutlu patch'lere böler ve embedding vektörlerine dönüştürür
- **Multi-Head Self-Attention**: Patch'ler arası ilişkileri öğrenir
- **MLP Blocks**: Feed-forward neural network katmanları
- **Transformer Encoder**: Attention ve MLP bloklarını birleştirir
- **Classification Head**: Final sınıflandırma katmanı

### 🔹 Teknik Detaylar
- **Patch Size**: 16x16 piksel
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Transformer Layers**: 12
- **MLP Size**: 3072

## 🚀 Kurulum ve Kullanım

### Gerekli Kütüphaneler

```bash
pip install torch torchvision matplotlib pandas tqdm pillow torchinfo
```

### 📁 Veri Yapısı

Veri setinizi aşağıdaki yapıda organize edin:

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

### 🏃‍♂️ Modelleri Çalıştırma

#### 1. Sıfırdan ViT Modeli
```bash
python image_classifier_from_scratch.py
```

#### 2. Önceden Eğitilmiş Model ile Transfer Learning
```bash
python train_using_pretrained_model_image_classifier.py
```

#### 3. Model Test ve Değerlendirme
```bash
python test_model.py
```

## 📊 Model Performansı

### 🎯 Değerlendirme Metrikleri
- **Genel Doğruluk Oranı**: Tüm test verisi üzerindeki başarı
- **Sınıf Bazında Doğruluk**: Her sınıf için ayrı performans analizi
- **Güven Skorları**: Her tahmin için güven aralığı
- **Confusion Matrix**: Detaylı hata analizi

### 📈 Çıktı Formatları
- **JSON**: Detaylı sonuçlar ve metadata
- **CSV**: Tabular veri analizi için
- **Görsel**: Tahmin örnekleri ve grafikler
- **Metin Raporu**: Özet istatistikler

## 🔧 Özelleştirme

### Model Hiperparametreleri
```python
# Eğitim parametreleri
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
IMG_SIZE = 224

# ViT parametreleri
patch_size = 16
embedding_dim = 768
num_heads = 12
num_transformer_layers = 12
```

### Veri Augmentasyonu
```python
transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])
```

## 📚 Kullanılan Teknolojiler

| Teknoloji | Versiyon | Açıklama |
|-----------|----------|----------|
| **PyTorch** | 2.x | Derin öğrenme framework'ü |
| **Torchvision** | 0.15+ | Görüntü işleme ve pretrained modeller |
| **Transformers** | Custom | Vision Transformer implementasyonu |
| **Matplotlib** | 3.x | Veri görselleştirme |
| **Pandas** | 1.x | Veri analizi |
| **PIL** | 8.x | Görüntü işleme |

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- **Attention Is All You Need** makalesinin yazarları
- **An Image is Worth 16x16 Words** makalesinin yazarları
- PyTorch ve Torchvision geliştiricileri
- Açık kaynak topluluğu

## 📞 İletişim

Proje hakkında sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

⭐ **Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!** ⭐
