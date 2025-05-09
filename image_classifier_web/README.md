# Görüntü Sınıflandırma Web Uygulaması

Bu Django tabanlı web uygulaması, önceden eğitilmiş görüntü sınıflandırma modellerini kullanarak görüntüleri sınıflandırmanıza olanak sağlar.

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install django pillow torch torchvision
```

2. Model dosyasını indirin:
- `animal_classifier_vit.pth` model dosyasını [buradan](MODEL_DOWNLOAD_LINK) indirin
- İndirdiğiniz model dosyasını `image_classifier_web/media/models/` klasörüne yerleştirin

3. Veri klasörünü hazırlayın:
- `image_classifier_web/yazlab-data/train/` klasörü oluşturun
- Her sınıf için bir alt klasör oluşturun (örn: kedi, köpek, kuş)
- Her sınıf klasörüne ilgili görüntüleri yerleştirin

4. Veritabanını oluşturun:
```bash
python manage.py makemigrations
python manage.py migrate
```

5. Sunucuyu başlatın:
```bash
python manage.py runserver
```

## Kullanım

1. Tarayıcınızda http://localhost:8000 adresine gidin
2. "Yeni Model Yükle" butonuna tıklayarak model yükleme sayfasına gidin
3. Model adı ve açıklama girin
4. Model dosyasını seçin ve yükleyin
5. Ana sayfaya dönün ve sınıflandırmak istediğiniz görüntüyü yükleyin

## Klasör Yapısı

```
image_classifier_web/
├── classifier/
├── yazlab-data/
│   └── train/
│       ├── kedi/
│       ├── kopek/
│       └── kus/
├── media/
│   └── models/
│       └── animal_classifier_vit.pth
├── manage.py
└── ...
```

## Notlar

- Model dosyası büyük olduğu için Git deposunda bulunmamaktadır
- Sınıf isimleri otomatik olarak `yazlab-data/train` klasöründeki alt klasör isimlerinden alınır
- Yüklenen görüntüler `media/uploads` klasöründe saklanır 