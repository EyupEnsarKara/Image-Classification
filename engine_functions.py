"""
PyTorch model eğitimi ve test fonksiyonları
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """PyTorch modelini tek bir epoch için eğitir.

    Hedef PyTorch modelini eğitim moduna alır ve sonra
    gerekli tüm eğitim adımlarını çalıştırır (ileri geçiş,
    kayıp hesaplama, optimizer adımı).

    Args:
        model: Eğitilecek PyTorch modeli.
        dataloader: Model üzerinde eğitim yapılacak DataLoader örneği.
        loss_fn: Minimize edilecek PyTorch kayıp fonksiyonu.
        optimizer: Kayıp fonksiyonunu minimize etmeye yardımcı PyTorch optimizer.
        device: Hesaplama yapılacak hedef cihaz (örn. "cuda" veya "cpu").

    Returns:
        Eğitim kaybı ve eğitim doğruluğu metriklerinin tuple'ı.
        (train_loss, train_accuracy) formunda. Örneğin:
        (0.1112, 0.8743)
    """
    # Modeli eğitim moduna al
    model.train()

    # Eğitim kaybı ve doğruluk değerlerini ayarla
    train_loss, train_acc = 0, 0

    # Veri yükleyici batch'leri üzerinde döngü
    for batch, (X, y) in enumerate(tqdm(dataloader, desc='Batch', leave=False)):
        # Veriyi hedef cihaza gönder
        X, y = X.to(device), y.to(device)

        # 1. İleri geçiş
        y_pred = model(X)

        # 2. Kaybı hesapla ve biriktir
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer sıfırla
        optimizer.zero_grad()

        # 4. Geri yayılım
        loss.backward()

        # 5. Optimizer adımı
        optimizer.step()

        # Tüm batch'ler boyunca doğruluk metriğini hesapla ve biriktir
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Batch başına ortalama kayıp ve doğruluk elde etmek için metrikleri ayarla
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """PyTorch modelini tek bir epoch için test eder.

    Hedef PyTorch modelini "eval" moduna alır ve sonra
    test veri seti üzerinde ileri geçiş yapar.

    Args:
        model: Test edilecek PyTorch modeli.
        dataloader: Model üzerinde test yapılacak DataLoader örneği.
        loss_fn: Test verisi üzerinde kayıp hesaplamak için PyTorch kayıp fonksiyonu.
        device: Hesaplama yapılacak hedef cihaz (örn. "cuda" veya "cpu").

    Returns:
        Test kaybı ve test doğruluğu metriklerinin tuple'ı.
        (test_loss, test_accuracy) formunda. Örneğin:
        (0.0223, 0.8985)
    """
    # Modeli değerlendirme moduna al
    model.eval() 

    # Test kaybı ve doğruluk değerlerini ayarla
    test_loss, test_acc = 0, 0

    # Çıkarım bağlam yöneticisini aç
    with torch.inference_mode():
        # DataLoader batch'leri üzerinde döngü
        for batch, (X, y) in enumerate(dataloader):
            # Veriyi hedef cihaza gönder
            X, y = X.to(device), y.to(device)

            # 1. İleri geçiş
            test_pred_logits = model(X)

            # 2. Kaybı hesapla ve biriktir
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Doğruluğu hesapla ve biriktir
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Batch başına ortalama kayıp ve doğruluk elde etmek için metrikleri ayarla
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train_model(model: torch.nn.Module, 
                train_dataloader: torch.utils.data.DataLoader, 
                test_dataloader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                epochs: int,
                device: torch.device,
                scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> Dict[str, List]:
    """PyTorch modelini eğitir ve test eder.

    Hedef PyTorch modelini train_step() ve test_step()
    fonksiyonları aracılığıyla belirli sayıda epoch için geçirir,
    aynı epoch döngüsünde modeli eğitir ve test eder.

    Değerlendirme metriklerini hesaplar, yazdırır ve saklar.

    Args:
        model: Eğitilecek ve test edilecek PyTorch modeli.
        train_dataloader: Model üzerinde eğitim yapılacak DataLoader örneği.
        test_dataloader: Model üzerinde test yapılacak DataLoader örneği.
        optimizer: Kayıp fonksiyonunu minimize etmeye yardımcı PyTorch optimizer.
        loss_fn: Her iki veri seti üzerinde kayıp hesaplamak için PyTorch kayıp fonksiyonu.
        scheduler: PyTorch öğrenme oranı zamanlayıcısı.
        epochs: Kaç epoch eğitim yapılacağını belirten tamsayı.
        device: Hesaplama yapılacak hedef cihaz (örn. "cuda" veya "cpu").

    Returns:
        Eğitim ve test kaybı ile eğitim ve test doğruluğu metriklerinin sözlüğü.
        Her metrik, her epoch için bir listede değer içerir.
        Şu formda: {train_loss: [...],
                   train_acc: [...],
                   test_loss: [...],
                   test_acc: [...]} 
        Örneğin epochs=2 için eğitim yapılırsa: 
                  {train_loss: [2.0616, 1.0537],
                   train_acc: [0.3945, 0.3945],
                   test_loss: [1.2641, 1.5706],
                   test_acc: [0.3400, 0.2973]} 
    """
    # Boş sonuçlar sözlüğü oluştur
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    # Modelin hedef cihazda olduğundan emin ol
    model.to(device)

    # Belirli sayıda epoch için eğitim ve test adımları üzerinde döngü
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)

        # Scheduler adımı
        if scheduler is not None:
            scheduler.step(test_loss)

        # Neler olduğunu yazdır
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Sonuçlar sözlüğünü güncelle
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Epoch'ların sonunda doldurulmuş sonuçları döndür
    return results 