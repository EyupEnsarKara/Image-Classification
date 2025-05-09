from django.urls import path
from . import views

urlpatterns = [
    path('', views.image_classification, name='image_classification'),
    path('upload-model/', views.upload_model, name='upload_model'),
] 