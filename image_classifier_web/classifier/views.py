from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from .models import UploadedImage, ModelFile
import torch
import torchvision
from PIL import Image
import os

def get_class_names():
    # yazlab-data/train klasörünü Django projesinin içinde ara
    train_dir = os.path.join(settings.BASE_DIR, 'yazlab-data', 'train')
    if os.path.exists(train_dir):
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        return class_names
    return []

def load_model(model_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.vit_b_16()
    model.heads = torch.nn.Linear(in_features=768, out_features=len(model_file.class_names))
    model.load_state_dict(torch.load(model_file.file.path, map_location=device))
    model.eval()
    return model, model_file.class_names

def process_image(image_path):
    transform = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def upload_model(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        model_file = request.FILES.get('model_file')
        
        if name and model_file:
            # Sınıf isimlerini otomatik olarak al
            class_names = get_class_names()
            if not class_names:
                messages.error(request, 'Sınıf isimleri bulunamadı! Lütfen yazlab-data/train klasörünü kontrol edin.')
                return redirect('upload_model')
            
            model = ModelFile(
                name=name,
                file=model_file,
                description=description,
                class_names=class_names
            )
            model.save()
            messages.success(request, 'Model başarıyla yüklendi!')
            return redirect('image_classification')
    
    return render(request, 'classifier/upload_model.html')

def image_classification(request):
    models = ModelFile.objects.all()
    
    if request.method == 'POST' and request.FILES.get('image'):
        model_id = request.POST.get('model')
        if not model_id:
            return render(request, 'classifier/classification.html', {
                'error': 'Lütfen bir model seçin',
                'models': models
            })
            
        try:
            model_file = ModelFile.objects.get(id=model_id)
            image_instance = UploadedImage(
                image=request.FILES['image'],
                model_used=model_file
            )
            image_instance.save()
            
            model, class_names = load_model(model_file)
            image_path = os.path.join(settings.MEDIA_ROOT, str(image_instance.image))
            processed_image = process_image(image_path)
            
            with torch.no_grad():
                output = model(processed_image)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = class_names[probabilities.argmax().item()]
                confidence = probabilities.max().item() * 100
                
            image_instance.prediction = f"{predicted_class} ({confidence:.2f}%)"
            image_instance.save()
            
            return render(request, 'classifier/classification.html', {
                'image_url': image_instance.image.url,
                'prediction': image_instance.prediction,
                'models': models,
                'selected_model': model_id
            })
            
        except Exception as e:
            return render(request, 'classifier/classification.html', {
                'error': f"Sınıflandırma hatası: {str(e)}",
                'models': models
            })
    
    return render(request, 'classifier/classification.html', {'models': models})
