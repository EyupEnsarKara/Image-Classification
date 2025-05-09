from django.db import models
import os

# Create your models here.

class ModelFile(models.Model):
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='models/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True)
    class_names = models.JSONField(default=list)  # Sınıf isimlerini JSON olarak saklayacağız
    
    def __str__(self):
        return self.name

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=100, blank=True)
    model_used = models.ForeignKey(ModelFile, on_delete=models.SET_NULL, null=True)
    
    def __str__(self):
        return f"Image uploaded at {self.uploaded_at}"
