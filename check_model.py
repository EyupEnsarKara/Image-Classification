"""
Model dosyasının yapısını kontrol eden yardımcı script
"""
import torch
import torchvision
import os

def check_model_structure():
    """Model dosyasının yapısını kontrol eder"""
    
    model_path = 'animal_classifier_vit.pth'
    
    if not os.path.exists(model_path):
        print("❌ Model dosyası bulunamadı!")
        return
    
    print("🔍 Model dosyası kontrol ediliyor...")
    
    try:
        # Model dosyasını yükle
        state_dict = torch.load(model_path, map_location='cpu')
        
        print(f"✅ Model dosyası başarıyla yüklendi")
        print(f"📊 Toplam katman sayısı: {len(state_dict)}")
        
        # Katman isimlerini ve boyutlarını göster
        print("\n📋 Model katmanları:")
        for name, tensor in state_dict.items():
            print(f"  {name}: {tensor.shape}")
        
        # Heads katmanını özel olarak kontrol et
        heads_layers = [name for name in state_dict.keys() if 'heads' in name]
        if heads_layers:
            print(f"\n🎯 Heads katmanları:")
            for layer in heads_layers:
                print(f"  {layer}: {state_dict[layer].shape}")
        
        # Sınıf sayısını tahmin et
        output_layers = [name for name in state_dict.keys() if 'head' in name and 'weight' in name]
        if output_layers:
            output_layer = output_layers[0]
            num_classes = state_dict[output_layer].shape[0]
            print(f"\n🔢 Tahmin edilen sınıf sayısı: {num_classes}")
        
        # Standart ViT modeli oluştur ve karşılaştır
        print(f"\n🔄 Standart ViT modeli ile karşılaştırma:")
        standard_model = torchvision.models.vit_b_16(weights=None)
        standard_dict = standard_model.state_dict()
        
        # Uyumlu ve uyumsuz katmanları bul
        compatible = []
        incompatible = []
        
        for name in state_dict.keys():
            if name in standard_dict:
                if state_dict[name].shape == standard_dict[name].shape:
                    compatible.append(name)
                else:
                    incompatible.append((name, state_dict[name].shape, standard_dict[name].shape))
            else:
                incompatible.append((name, state_dict[name].shape, "Standart modelde yok"))
        
        print(f"✅ Uyumlu katmanlar: {len(compatible)}")
        print(f"❌ Uyumsuz katmanlar: {len(incompatible)}")
        
        if incompatible:
            print(f"\n⚠️ Uyumsuz katmanlar:")
            for name, saved_shape, expected_shape in incompatible:
                print(f"  {name}: kaydedilen={saved_shape}, beklenen={expected_shape}")
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")

if __name__ == "__main__":
    check_model_structure() 