from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from .models import UploadedImage, ModelFile
import torch
import torchvision
from PIL import Image
import os
import zipfile
import tempfile
import json
import uuid
import shutil
import datetime

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

def classify_single_image(model, class_names, image_path):
    try:
        processed_image = process_image(image_path)
        
        with torch.no_grad():
            output = model(processed_image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Tüm sınıfların tahminlerini diziye ekle
            all_predictions = []
            for i, prob in enumerate(probabilities):
                all_predictions.append({
                    "class": class_names[i],
                    "confidence": round(prob.item() * 100, 2)
                })
            
            # En yüksek olasılıklı sınıfı bul
            max_index = probabilities.argmax().item()
            predicted_class = class_names[max_index]
            confidence = probabilities[max_index].item() * 100
            
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "all_predictions": sorted(all_predictions, key=lambda x: x["confidence"], reverse=True)
        }
    except Exception as e:
        return {"error": str(e)}

def image_classification(request):
    models = ModelFile.objects.all()
    
    # JSON yanıtı isteniyor mu kontrol et
    want_json = request.GET.get('json', False) or request.POST.get('json', False)
    download_json = request.GET.get('download', False) or request.POST.get('download', False)
    
    if request.method == 'POST':
        model_id = request.POST.get('model')
        
        if not model_id:
            error_message = 'Lütfen bir model seçin'
            if want_json:
                return JsonResponse({"error": error_message}, status=400)
            return render(request, 'classifier/classification.html', {
                'error': error_message,
                'models': models
            })
            
        try:
            model_file = ModelFile.objects.get(id=model_id)
            model, class_names = load_model(model_file)
            
            # Zip dosyası kontrolü
            if request.FILES.get('zip_file'):
                zip_file = request.FILES['zip_file']
                temp_dir = tempfile.mkdtemp()
                results = []
                
                try:
                    # Zip dosyasını geçici dizine çıkar
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Her bir resmi sınıflandır
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_path = os.path.join(root, file)
                                result = classify_single_image(model, class_names, image_path)
                                result['filename'] = file
                                results.append(result)
                    
                    json_data = {
                        "zip_analysis": results,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                                
                    if want_json:
                        if download_json:
                            response = HttpResponse(
                                json.dumps(json_data, indent=4),
                                content_type='application/json'
                            )
                            response['Content-Disposition'] = 'attachment; filename=zip_analysis_results.json'
                            return response
                        return JsonResponse(json_data)
                        
                    return render(request, 'classifier/classification.html', {
                        'zip_results': results,
                        'models': models,
                        'selected_model': model_id,
                        'json_data': json.dumps(json_data, indent=4)
                    })
                finally:
                    # Geçici dizini temizle
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Tek resim işleme
            elif request.FILES.get('image'):
                image_instance = UploadedImage(
                    image=request.FILES['image'],
                    model_used=model_file
                )
                image_instance.save()
                
                image_path = os.path.join(settings.MEDIA_ROOT, str(image_instance.image))
                result = classify_single_image(model, class_names, image_path)
                
                if "error" not in result:
                    prediction_text = f"{result['predicted_class']} ({result['confidence']}%)"
                    image_instance.prediction = prediction_text
                    image_instance.save()
                    
                    json_data = {
                        "image_url": request.build_absolute_uri(image_instance.image.url),
                        "prediction": result,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    if want_json:
                        if download_json:
                            response = HttpResponse(
                                json.dumps(json_data, indent=4),
                                content_type='application/json'
                            )
                            filename = f"image_analysis_{image_instance.image.name.split('/')[-1].split('.')[0]}.json"
                            response['Content-Disposition'] = f'attachment; filename={filename}'
                            return response
                        return JsonResponse(json_data)
                    
                    return render(request, 'classifier/classification.html', {
                        'image_url': image_instance.image.url,
                        'prediction': prediction_text,
                        'predictions_list': result['all_predictions'],
                        'models': models,
                        'selected_model': model_id,
                        'json_data': json.dumps(json_data, indent=4)
                    })
                else:
                    if want_json:
                        return JsonResponse({"error": result["error"]}, status=500)
            else:
                error_message = 'Lütfen bir resim veya zip dosyası yükleyin'
                if want_json:
                    return JsonResponse({"error": error_message}, status=400)
                return render(request, 'classifier/classification.html', {
                    'error': error_message,
                    'models': models
                })
                
        except Exception as e:
            error_message = f"Sınıflandırma hatası: {str(e)}"
            if want_json:
                return JsonResponse({"error": error_message}, status=500)
            return render(request, 'classifier/classification.html', {
                'error': error_message,
                'models': models
            })
    
    if want_json:
        return JsonResponse({"message": "POST isteği bekleniyor"})
    return render(request, 'classifier/classification.html', {'models': models})
