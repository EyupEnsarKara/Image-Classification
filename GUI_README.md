# 🤖 AI Model Test Arayüzü Kullanım Kılavuzu

Bu proje, eğitilmiş AI modelinizi test etmek için iki farklı modern arayüz sunar.

## 📋 Özellikler

- ✨ Modern ve kullanıcı dostu arayüz
- 🖼️ Görüntü seçme ve önizleme
- 🔮 Gerçek zamanlı tahmin yapma
- 📊 Güven skoru gösterimi
- 🎯 Detaylı sonuç analizi
- 💻 GPU/CPU otomatik algılama
- 🔄 İlerleme çubuğu ile durum takibi
- 🔧 Otomatik model uyumluluk kontrolü

## 🚀 Kurulum

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Ek Kütüphane (CustomTkinter için)
```bash
pip install customtkinter
```

## 📱 Arayüz Seçenekleri

### 1. Modern Arayüz (CustomTkinter)
```bash
python model_test_gui.py
```

**Özellikler:**
- 🎨 Modern dark theme
- 🔄 Smooth animasyonlar
- 📱 Responsive tasarım
- 🎯 Gelişmiş UI bileşenleri

### 2. Basit Arayüz (Standart Tkinter)
```bash
python simple_model_gui.py
```

**Özellikler:**
- 🖥️ Standart tkinter kullanır
- 🚀 Hızlı başlatma
- 💾 Düşük sistem gereksinimi
- 🔧 Kolay özelleştirme

## 🎮 Kullanım

### Adım 1: Model Yükleme
- Uygulama başlatıldığında model otomatik yüklenir
- Model durumu sol panelde gösterilir
- ✅ yeşil işaret = Model hazır
- ❌ kırmızı işaret = Model hatası

### Adım 2: Görüntü Seçme
1. **"📁 Görüntü Seç"** butonuna tıklayın
2. Desteklenen formatlar: JPG, JPEG, PNG, BMP, GIF
3. Seçilen görüntü sağ panelde görüntülenir

### Adım 3: Tahmin Yapma
1. **"🔮 Tahmin Yap"** butonuna tıklayın
2. İlerleme çubuğu tahmin sürecini gösterir
3. Sonuçlar sol panelde görüntülenir

## 📊 Sonuç Yorumlama

### Tahmin Sonuçları
- **🎯 Tahmin:** Modelin öngördüğü sınıf
- **📊 Güven:** Tahmin güvenilirlik yüzdesi
- **✅ Durum:** İşlem durumu bilgisi

### Güven Skoru Rehberi
- **%90-100:** Çok yüksek güven
- **%70-89:** Yüksek güven  
- **%50-69:** Orta güven
- **%0-49:** Düşük güven

## 🛠️ Teknik Detaylar

### Model Gereksinimleri
- Model dosyası: `animal_classifier_vit.pth`
- Model tipi: Vision Transformer (ViT)
- Sınıf sayısı: Otomatik algılanır (model dosyasından)

### Otomatik Model Uyumluluk
Uygulama şu özelliklere sahiptir:
- 🔍 Model dosyasından sınıf sayısını otomatik algılar
- 🔧 Farklı model yapılarını destekler (`heads.weight` vs `heads.head.weight`)
- ⚡ Uyumsuz katmanları otomatik atlar
- 📝 Sınıf isimlerini otomatik oluşturur

### Desteklenen Model Yapıları
- **Standart ViT:** `heads.head.weight` ve `heads.head.bias`
- **Özel ViT:** `heads.weight` ve `heads.bias`
- **Herhangi bir sınıf sayısı:** 10, 90, 1000+ sınıf desteklenir

### Desteklenen Hayvan Sınıfları (90 Sınıf)
Model şu hayvan türlerini tanıyabilir:

**🐾 Memeli Hayvanlar:**
antelope, badger, bat, bear, bison, boar, cat, chimpanzee, cow, coyote, deer, dog, dolphin, donkey, elephant, fox, goat, gorilla, hamster, hare, hedgehog, hippopotamus, horse, hyena, kangaroo, koala, leopard, lion, lizard, mouse, okapi, orangutan, otter, ox, panda, pig, porcupine, possum, raccoon, rat, reindeer, rhinoceros, seal, sheep, squirrel, tiger, whale, wolf, wombat, zebra

**🐦 Kuşlar:**
crow, duck, eagle, flamingo, goose, hornbill, hummingbird, owl, parrot, pelecaniformes, penguin, pigeon, sandpiper, sparrow, swan, turkey, woodpecker

**🐛 Böcekler:**
bee, beetle, butterfly, caterpillar, cockroach, dragonfly, fly, grasshopper, ladybugs, mosquito, moth

**🐠 Deniz Canlıları:**
goldfish, jellyfish, lobster, octopus, oyster, seahorse, shark, squid, starfish

**🐢 Sürüngenler:**
snake, turtle

**🐸 Amfibiler:**
crab

### Görüntü İşleme
- **Boyut:** 224x224 piksel
- **Normalizasyon:** ImageNet standartları
- **Format:** RGB renk uzayı

## 🔧 Sorun Giderme

### Model Yükleme Hataları
```
❌ Model dosyası bulunamadı!
```
**Çözüm:** `animal_classifier_vit.pth` dosyasının proje klasöründe olduğundan emin olun.

### Model Boyut Uyumsuzluğu
```
Error(s) in loading state_dict for VisionTransformer
```
**Çözüm:** 
- Uygulama otomatik olarak uyumlu katmanları yükler
- Konsol çıktısını kontrol edin
- Model dosyasının bozuk olmadığından emin olun

### Görüntü Yükleme Hataları
```
Görüntü yüklenirken hata oluştu
```
**Çözüm:** 
- Desteklenen format kullanın (JPG, PNG, etc.)
- Dosya boyutunu kontrol edin
- Dosya yolunda Türkçe karakter olmamasına dikkat edin

### Tahmin Hataları
```
Tahmin yapılırken hata oluştu
```
**Çözüm:**
- GPU belleği yeterli mi kontrol edin
- Model dosyası bozuk olabilir
- Görüntü formatını kontrol edin

## 🔍 Model Kontrol Aracı

Model dosyanızın yapısını kontrol etmek için:
```bash
python check_model.py
```

Bu araç şunları gösterir:
- 📊 Model katman sayısı
- 🎯 Sınıf sayısı
- ⚙️ Model yapısı uyumluluğu
- 🔧 Olası sorunlar

## 🎨 Özelleştirme

### Renk Teması Değiştirme
`simple_model_gui.py` dosyasında `colors` sözlüğünü düzenleyin:

```python
self.colors = {
    'bg': '#2b2b2b',        # Arka plan
    'fg': '#ffffff',        # Metin rengi
    'accent': '#0078d4',    # Vurgu rengi
    'success': '#107c10',   # Başarı rengi
    'warning': '#ff8c00',   # Uyarı rengi
    'error': '#d13438',     # Hata rengi
    'card': '#3c3c3c'       # Kart arka planı
}
```

### Pencere Boyutu
```python
self.root.geometry("1000x700")  # Genişlik x Yükseklik
```

## 📞 Destek

Herhangi bir sorun yaşarsanız:
1. `python check_model.py` ile model dosyasını kontrol edin
2. Konsol çıktısını kontrol edin
3. Model dosyasının varlığını doğrulayın
4. Gerekli kütüphanelerin yüklü olduğundan emin olun

## 🔄 Güncellemeler

### v1.0
- ✅ Temel arayüz
- ✅ Model yükleme
- ✅ Görüntü tahmin

### v1.1 (Mevcut)
- ✅ Otomatik model uyumluluk kontrolü
- ✅ Farklı model yapıları desteği
- ✅ Sınıf sayısı otomatik algılama
- ✅ Model kontrol aracı

### v1.2 (Gelecek)
- 🔄 Batch tahmin
- 📊 Detaylı metrikler
- 💾 Sonuç kaydetme
- 🎨 Tema seçenekleri

---

**🎯 İyi tahminler!** 🚀 