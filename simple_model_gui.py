"""
Basit Tkinter Arayüzü ile Model Test Uygulaması
CustomTkinter gerektirmez, standart tkinter kullanır
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torchvision
from PIL import Image, ImageTk
import os
import numpy as np
from torchvision import transforms
import threading
import zipfile
import io

class SimpleModelTestGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 AI Model Test Arayüzü - Detaylı Analiz")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Modern renkler
        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'accent': '#0078d4',
            'success': '#107c10',
            'warning': '#ff8c00',
            'error': '#d13438',
            'card': '#3c3c3c'
        }
        
        # Model ve sınıf isimleri
        self.model = None
        self.class_names = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_image_path = None
        self.last_predictions = None  # Son tahmin sonuçları
        
        # ZIP dosyası desteği
        self.current_zip_file = None
        self.zip_images = []  # ZIP içindeki görüntü listesi
        self.current_zip_image = None  # Seçili ZIP görüntüsü
        
        # Toplu test desteği
        self.batch_results = {}  # Toplu test sonuçları
        self.is_batch_testing = False
        
        # Stil ayarları
        self.setup_styles()
        
        # Arayüzü oluştur
        self.create_widgets()
        self.load_model()
        
    def setup_styles(self):
        """Modern stil ayarları"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Buton stilleri
        style.configure('Modern.TButton',
                       background=self.colors['accent'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10, 'bold'))
        
        style.map('Modern.TButton',
                 background=[('active', '#106ebe')])
        
        # Çerçeve stilleri
        style.configure('Card.TFrame',
                       background=self.colors['card'],
                       relief='flat',
                       borderwidth=1)
        
        # Label stilleri
        style.configure('Title.TLabel',
                       background=self.colors['bg'],
                       foreground=self.colors['fg'],
                       font=('Segoe UI', 16, 'bold'))
        
        style.configure('Modern.TLabel',
                       background=self.colors['card'],
                       foreground=self.colors['fg'],
                       font=('Segoe UI', 10))
        
        # Treeview stilleri
        style.configure('Modern.Treeview',
                       background=self.colors['card'],
                       foreground=self.colors['fg'],
                       fieldbackground=self.colors['card'],
                       borderwidth=0)
        
        style.configure('Modern.Treeview.Heading',
                       background=self.colors['accent'],
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'))
        
    def create_widgets(self):
        """Modern arayüz bileşenlerini oluşturur"""
        
        # Ana başlık
        title_frame = tk.Frame(self.root, bg=self.colors['bg'], height=60)
        title_frame.pack(fill='x', padx=20, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🤖 AI Model Test Arayüzü - Detaylı Analiz",
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Segoe UI', 18, 'bold')
        )
        title_label.pack(expand=True)
        
        # Ana içerik çerçevesi
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Sol panel - Kontroller
        left_panel = tk.Frame(main_frame, bg=self.colors['card'], width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Sol panel içeriği
        self.create_control_panel(left_panel)
        
        # Orta panel - Görüntü
        middle_panel = tk.Frame(main_frame, bg=self.colors['card'], width=400)
        middle_panel.pack(side='left', fill='y', padx=(0, 10))
        middle_panel.pack_propagate(False)
        
        # Orta panel içeriği
        self.create_image_panel(middle_panel)
        
        # Sağ panel - Detaylı sonuçlar
        right_panel = tk.Frame(main_frame, bg=self.colors['card'])
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Sağ panel içeriği
        self.create_results_panel(right_panel)
        
        # Alt durum çubuğu
        self.create_status_bar()
        
    def create_control_panel(self, parent):
        """Kontrol panelini oluşturur"""
        
        # Panel başlığı
        control_title = tk.Label(
            parent,
            text="🎛️ Kontrol Paneli",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 14, 'bold')
        )
        control_title.pack(pady=15)
        
        # Model durumu
        status_frame = tk.Frame(parent, bg=self.colors['card'])
        status_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(
            status_frame,
            text="📊 Model Durumu:",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor='w')
        
        self.model_status_label = tk.Label(
            status_frame,
            text="Yükleniyor...",
            bg=self.colors['card'],
            fg=self.colors['warning'],
            font=('Segoe UI', 9)
        )
        self.model_status_label.pack(anchor='w', pady=(5, 0))
        
        # Cihaz bilgisi
        device_frame = tk.Frame(parent, bg=self.colors['card'])
        device_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(
            device_frame,
            text="💻 Cihaz:",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor='w')
        
        tk.Label(
            device_frame,
            text=self.device.upper(),
            bg=self.colors['card'],
            fg=self.colors['accent'],
            font=('Segoe UI', 9)
        ).pack(anchor='w', pady=(5, 0))
        
        # Butonlar
        button_frame = tk.Frame(parent, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=20, pady=20)
        
        self.select_image_btn = tk.Button(
            button_frame,
            text="📁 Görüntü Seç",
            command=self.select_image,
            bg=self.colors['accent'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            cursor='hand2',
            height=2
        )
        self.select_image_btn.pack(fill='x', pady=(0, 10))
        
        self.select_zip_btn = tk.Button(
            button_frame,
            text="📦 ZIP Dosyası Seç",
            command=self.select_zip_file,
            bg=self.colors['warning'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            cursor='hand2',
            height=2
        )
        self.select_zip_btn.pack(fill='x', pady=(0, 10))
        
        self.predict_btn = tk.Button(
            button_frame,
            text="🔮 Tahmin Yap",
            command=self.predict_image,
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            cursor='hand2',
            height=2,
            state='disabled'
        )
        self.predict_btn.pack(fill='x')
        
        # Hızlı özet
        summary_frame = tk.Frame(parent, bg=self.colors['card'])
        summary_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            summary_frame,
            text="📈 Hızlı Özet",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=(0, 10))
        
        # En yüksek tahmin
        self.top_prediction_label = tk.Label(
            summary_frame,
            text="Henüz tahmin yapılmadı",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 10),
            wraplength=250
        )
        self.top_prediction_label.pack(pady=5)
        
        # Güven skoru
        self.confidence_label = tk.Label(
            summary_frame,
            text="Güven: -",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 10)
        )
        self.confidence_label.pack(pady=5)
        
        # İlerleme çubuğu
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            summary_frame,
            variable=self.progress_var,
            maximum=100,
            length=250
        )
        self.progress_bar.pack(pady=15)
        
        # ZIP görüntü listesi
        zip_frame = tk.Frame(parent, bg=self.colors['card'])
        zip_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        self.zip_label = tk.Label(
            zip_frame,
            text="📦 ZIP Dosyası Seçilmedi",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 12, 'bold')
        )
        self.zip_label.pack(pady=(10, 5))
        
        # Listbox ve scrollbar (başlangıçta gizli)
        listbox_frame = tk.Frame(zip_frame, bg=self.colors['card'])
        
        self.zip_listbox = tk.Listbox(
            listbox_frame,
            bg='white',
            fg='black',
            font=('Segoe UI', 9),
            height=6,
            selectmode='single'
        )
        self.zip_listbox.bind('<<ListboxSelect>>', self.on_zip_image_select)
        
        zip_scrollbar = ttk.Scrollbar(listbox_frame, orient='vertical', command=self.zip_listbox.yview)
        self.zip_listbox.configure(yscrollcommand=zip_scrollbar.set)
        
        self.zip_listbox.pack(side='left', fill='both', expand=True)
        zip_scrollbar.pack(side='right', fill='y')
        
        # Toplu test butonu (başlangıçta gizli)
        self.batch_test_btn = tk.Button(
            zip_frame,
            text="🚀 Tüm Görüntüleri Test Et",
            command=self.batch_test_images,
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            cursor='hand2',
            height=1,
            state='disabled'
        )
        
        # Başlangıçta listbox ve butonu gizle
        self.listbox_frame = listbox_frame
        self.zip_frame = zip_frame
        
    def create_image_panel(self, parent):
        """Görüntü panelini oluşturur"""
        
        # Panel başlığı
        image_title = tk.Label(
            parent,
            text="🖼️ Seçilen Görüntü",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 14, 'bold')
        )
        image_title.pack(pady=15)
        
        # Görüntü çerçevesi
        image_frame = tk.Frame(parent, bg=self.colors['bg'], relief='sunken', bd=2)
        image_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        self.image_label = tk.Label(
            image_frame,
            text="Görüntü seçilmedi\n\n📁 Lütfen bir görüntü seçin",
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Segoe UI', 12),
            justify='center'
        )
        self.image_label.pack(expand=True)
        
    def create_results_panel(self, parent):
        """Detaylı sonuçlar panelini oluşturur"""
        
        # Panel başlığı
        results_title = tk.Label(
            parent,
            text="📊 Detaylı Tahmin Sonuçları",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 14, 'bold')
        )
        results_title.pack(pady=15)
        
        # Notebook (sekmeli panel)
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Sekme 1: En yüksek tahminler
        self.top_frame = tk.Frame(self.notebook, bg=self.colors['card'])
        self.notebook.add(self.top_frame, text="🏆 En Yüksek 10")
        
        # Sekme 2: Tüm sonuçlar
        self.all_frame = tk.Frame(self.notebook, bg=self.colors['card'])
        self.notebook.add(self.all_frame, text="📋 Tüm Sonuçlar")
        
        # En yüksek 10 tahmin listesi
        self.create_top_predictions_list()
        
        # Tüm sonuçlar listesi
        self.create_all_predictions_list()
        
    def create_top_predictions_list(self):
        """En yüksek 10 tahmin listesini oluşturur"""
        
        # Başlık
        top_title = tk.Label(
            self.top_frame,
            text="🏆 En Yüksek 10 Tahmin",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 12, 'bold')
        )
        top_title.pack(pady=10)
        
        # Treeview
        columns = ('Sıra', 'Hayvan', 'Olasılık', 'Yüzde')
        self.top_tree = ttk.Treeview(
            self.top_frame, 
            columns=columns, 
            show='headings',
            style='Modern.Treeview',
            height=10
        )
        
        # Sütun başlıkları
        self.top_tree.heading('Sıra', text='🥇 Sıra')
        self.top_tree.heading('Hayvan', text='🐾 Hayvan')
        self.top_tree.heading('Olasılık', text='📊 Olasılık')
        self.top_tree.heading('Yüzde', text='📈 Yüzde')
        
        # Sütun genişlikleri
        self.top_tree.column('Sıra', width=60, anchor='center')
        self.top_tree.column('Hayvan', width=120, anchor='w')
        self.top_tree.column('Olasılık', width=80, anchor='center')
        self.top_tree.column('Yüzde', width=80, anchor='center')
        
        # Scrollbar
        top_scrollbar = ttk.Scrollbar(self.top_frame, orient='vertical', command=self.top_tree.yview)
        self.top_tree.configure(yscrollcommand=top_scrollbar.set)
        
        # Pack
        self.top_tree.pack(side='left', fill='both', expand=True, padx=(20, 0), pady=10)
        top_scrollbar.pack(side='right', fill='y', padx=(0, 20), pady=10)
        
    def create_all_predictions_list(self):
        """Tüm tahminler listesini oluşturur"""
        
        # Başlık ve arama
        search_frame = tk.Frame(self.all_frame, bg=self.colors['card'])
        search_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(
            search_frame,
            text="🔍 Hayvan Ara:",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 10, 'bold')
        ).pack(side='left')
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_predictions)
        search_entry = tk.Entry(
            search_frame,
            textvariable=self.search_var,
            font=('Segoe UI', 10),
            bg='white',
            fg='black'
        )
        search_entry.pack(side='left', padx=(10, 0), fill='x', expand=True)
        
        # Treeview
        columns = ('Hayvan', 'Olasılık', 'Yüzde')
        self.all_tree = ttk.Treeview(
            self.all_frame, 
            columns=columns, 
            show='headings',
            style='Modern.Treeview'
        )
        
        # Sütun başlıkları
        self.all_tree.heading('Hayvan', text='🐾 Hayvan')
        self.all_tree.heading('Olasılık', text='📊 Olasılık')
        self.all_tree.heading('Yüzde', text='📈 Yüzde')
        
        # Sütun genişlikleri
        self.all_tree.column('Hayvan', width=150, anchor='w')
        self.all_tree.column('Olasılık', width=100, anchor='center')
        self.all_tree.column('Yüzde', width=100, anchor='center')
        
        # Scrollbar
        all_scrollbar = ttk.Scrollbar(self.all_frame, orient='vertical', command=self.all_tree.yview)
        self.all_tree.configure(yscrollcommand=all_scrollbar.set)
        
        # Pack
        self.all_tree.pack(side='left', fill='both', expand=True, padx=(20, 0), pady=10)
        all_scrollbar.pack(side='right', fill='y', padx=(0, 20), pady=10)
        
    def create_status_bar(self):
        """Alt durum çubuğunu oluşturur"""
        status_frame = tk.Frame(self.root, bg=self.colors['accent'], height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="✅ Hazır - Görüntü seçin ve tahmin yapın",
            bg=self.colors['accent'],
            fg='white',
            font=('Segoe UI', 9)
        )
        self.status_label.pack(expand=True)
    
    def filter_predictions(self, *args):
        """Tahmin listesini filtreler"""
        if not self.last_predictions:
            return
            
        search_term = self.search_var.get().lower()
        
        # Tüm öğeleri temizle
        for item in self.all_tree.get_children():
            self.all_tree.delete(item)
        
        # Filtrelenmiş sonuçları ekle
        for i, (class_name, probability) in enumerate(self.last_predictions):
            if search_term in class_name.lower():
                percentage = f"%{probability*100:.2f}"
                prob_str = f"{probability:.4f}"
                
                self.all_tree.insert('', 'end', values=(
                    class_name.title(), 
                    prob_str, 
                    percentage
                ))
    
    def update_predictions_display(self, probabilities):
        """Tahmin sonuçlarını görsel olarak günceller"""
        
        # Olasılıkları sırala
        sorted_predictions = []
        for i, prob in enumerate(probabilities[0]):
            sorted_predictions.append((self.class_names[i], prob.item()))
        
        sorted_predictions.sort(key=lambda x: x[1], reverse=True)
        self.last_predictions = sorted_predictions
        
        # En yüksek 10 tahmin listesini güncelle
        self.update_top_predictions(sorted_predictions[:10])
        
        # Tüm sonuçlar listesini güncelle
        self.update_all_predictions(sorted_predictions)
    
    def update_top_predictions(self, top_predictions):
        """En yüksek tahminler listesini günceller"""
        
        # Mevcut öğeleri temizle
        for item in self.top_tree.get_children():
            self.top_tree.delete(item)
        
        # Yeni öğeleri ekle
        for i, (class_name, probability) in enumerate(top_predictions, 1):
            percentage = f"%{probability*100:.2f}"
            prob_str = f"{probability:.4f}"
            
            # Renk kodlaması için tag
            if i == 1:
                tag = 'gold'
            elif i == 2:
                tag = 'silver'
            elif i == 3:
                tag = 'bronze'
            else:
                tag = 'normal'
            
            self.top_tree.insert('', 'end', values=(
                f"{i}.", 
                class_name.title(), 
                prob_str, 
                percentage
            ), tags=(tag,))
        
        # Tag renklerini ayarla
        self.top_tree.tag_configure('gold', background='#FFD700', foreground='black')
        self.top_tree.tag_configure('silver', background='#C0C0C0', foreground='black')
        self.top_tree.tag_configure('bronze', background='#CD7F32', foreground='white')
        self.top_tree.tag_configure('normal', background=self.colors['card'], foreground=self.colors['fg'])
    
    def update_all_predictions(self, all_predictions):
        """Tüm tahminler listesini günceller"""
        
        # Mevcut öğeleri temizle
        for item in self.all_tree.get_children():
            self.all_tree.delete(item)
        
        # Yeni öğeleri ekle
        for class_name, probability in all_predictions:
            percentage = f"%{probability*100:.2f}"
            prob_str = f"{probability:.4f}"
            
            self.all_tree.insert('', 'end', values=(
                class_name.title(), 
                prob_str, 
                percentage
            ))
    
    def load_model(self):
        """Eğitilmiş modeli yükler"""
        try:
            # Model dosyasının varlığını kontrol et
            model_path = 'animal_classifier_vit.pth'
            if not os.path.exists(model_path):
                self.model_status_label.configure(
                    text="❌ Model dosyası bulunamadı!",
                    fg=self.colors['error']
                )
                self.status_label.configure(text="❌ Hata: Model dosyası bulunamadı")
                return
            
            # Model dosyasından sınıf sayısını al
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Heads katmanından sınıf sayısını belirle
            if 'heads.weight' in state_dict:
                num_classes = state_dict['heads.weight'].shape[0]
            elif 'heads.head.weight' in state_dict:
                num_classes = state_dict['heads.head.weight'].shape[0]
            else:
                # Varsayılan sınıf isimleri
                self.class_names = ['cat', 'dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cow', 'sheep', 'spider', 'squirrel']
                num_classes = len(self.class_names)
            
            # Sınıf isimlerini yükle veya oluştur
            val_dir = 'yazlab-data/val'
            if os.path.exists(val_dir):
                available_classes = sorted(os.listdir(val_dir))
                if len(available_classes) == num_classes:
                    self.class_names = available_classes
                else:
                    # Eğer sınıf sayısı uyuşmuyorsa, gerçek hayvan sınıflarını kullan
                    self.class_names = [
                        "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat",
                        "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", "dolphin",
                        "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish",
                        "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird",
                        "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito",
                        "moth", "mouse", "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda",
                        "parrot", "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer",
                        "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel",
                        "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
                    ][:num_classes]  # Sınıf sayısına göre kırp
            else:
                # Gerçek hayvan sınıflarını kullan
                self.class_names = [
                    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat",
                    "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", "dolphin",
                    "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish",
                    "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird",
                    "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito",
                    "moth", "mouse", "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda",
                    "parrot", "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer",
                    "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel",
                    "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
                ][:num_classes]  # Sınıf sayısına göre kırp
            
            print(f"Sınıf sayısı: {num_classes}")
            print(f"Sınıf isimleri: {self.class_names}")
            
            # Modeli oluştur
            self.model = torchvision.models.vit_b_16(weights=None)
            
            # Model başlığını doğru boyutla ayarla
            self.model.heads.head = torch.nn.Linear(in_features=768, out_features=num_classes)
            
            # Model ağırlıklarını yükle
            try:
                # Eğer model dosyasında heads.weight varsa, heads.head.weight'e dönüştür
                if 'heads.weight' in state_dict and 'heads.head.weight' not in state_dict:
                    state_dict['heads.head.weight'] = state_dict.pop('heads.weight')
                    state_dict['heads.head.bias'] = state_dict.pop('heads.bias')
                
                self.model.load_state_dict(state_dict)
                print("Model ağırlıkları başarıyla yüklendi")
                
            except RuntimeError as e:
                print(f"Boyut uyumsuzluğu tespit edildi: {e}")
                # Sadece uyumlu katmanları yükle
                model_dict = self.model.state_dict()
                
                # Uyumlu katmanları filtrele
                filtered_dict = {}
                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        filtered_dict[k] = v
                    else:
                        print(f"Atlanan katman: {k}")
                
                # Filtrelenmiş ağırlıkları yükle
                model_dict.update(filtered_dict)
                self.model.load_state_dict(model_dict)
                print("Uyumlu katmanlar yüklendi")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.model_status_label.configure(
                text="✅ Model başarıyla yüklendi!",
                fg=self.colors['success']
            )
            self.status_label.configure(text="✅ Model hazır - Görüntü seçebilirsiniz")
            
        except Exception as e:
            self.model_status_label.configure(
                text="❌ Model yükleme hatası!",
                fg=self.colors['error']
            )
            self.status_label.configure(text=f"❌ Hata: {str(e)}")
            messagebox.showerror("Hata", f"Model yüklenirken hata oluştu:\n{str(e)}")
    
    def select_image(self):
        """Görüntü seçme dialog'unu açar"""
        file_types = [
            ("Görüntü dosyaları", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("JPEG dosyaları", "*.jpg *.jpeg"),
            ("PNG dosyaları", "*.png"),
            ("Tüm dosyalar", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Tahmin için görüntü seçin",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.current_zip_image = None  # ZIP seçimini sıfırla
            self.display_image(file_path)
            self.predict_btn.configure(state='normal')
            self.status_label.configure(text="✅ Görüntü seçildi - Tahmin yapabilirsiniz")
            
            # Sonuçları sıfırla
            self.top_prediction_label.configure(text="Henüz tahmin yapılmadı")
            self.confidence_label.configure(text="Güven: -")
            self.progress_var.set(0)
    
    def select_zip_file(self):
        """ZIP dosyası seçme dialog'unu açar"""
        file_types = [
            ("ZIP dosyaları", "*.zip"),
            ("Tüm dosyalar", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Görüntü içeren ZIP dosyası seçin",
            filetypes=file_types
        )
        
        if file_path:
            try:
                self.load_zip_images(file_path)
                self.status_label.configure(text="✅ ZIP dosyası yüklendi - Bir görüntü seçin")
            except Exception as e:
                messagebox.showerror("Hata", f"ZIP dosyası yüklenirken hata oluştu:\n{str(e)}")
    
    def load_zip_images(self, zip_path):
        """ZIP dosyasındaki görüntüleri yükler"""
        self.current_zip_file = zip_path
        self.zip_images = []
        
        # Desteklenen görüntü formatları
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            for file_info in zip_file.filelist:
                if not file_info.is_dir():
                    file_ext = os.path.splitext(file_info.filename)[1].lower()
                    if file_ext in image_extensions:
                        self.zip_images.append(file_info.filename)
        
        # Listbox'ı güncelle
        self.zip_listbox.delete(0, tk.END)
        for image_name in self.zip_images:
            # Sadece dosya adını göster (klasör yolunu kırp)
            display_name = os.path.basename(image_name)
            self.zip_listbox.insert(tk.END, display_name)
        
        # ZIP frame'ini göster
        if self.zip_images:
            # Listbox ve butonu göster
            self.listbox_frame.pack(fill='x', padx=10, pady=(0, 10))
            self.batch_test_btn.pack(pady=(0, 10), padx=10, fill='x')
            
            self.zip_label.configure(text=f"📦 ZIP Görüntüleri ({len(self.zip_images)} adet)")
            self.batch_test_btn.configure(state='normal')
        else:
            messagebox.showwarning("Uyarı", "ZIP dosyasında görüntü bulunamadı!")
    
    def on_zip_image_select(self, event):
        """ZIP listesinden görüntü seçildiğinde çalışır"""
        selection = self.zip_listbox.curselection()
        if selection:
            index = selection[0]
            self.current_zip_image = self.zip_images[index]
            self.current_image_path = None  # Normal dosya seçimini sıfırla
            
            try:
                self.display_zip_image(self.current_zip_image)
                self.predict_btn.configure(state='normal')
                image_name = os.path.basename(self.current_zip_image)
                self.status_label.configure(text=f"✅ ZIP görüntüsü seçildi: {image_name}")
                
                # Sonuçları sıfırla
                self.top_prediction_label.configure(text="Henüz tahmin yapılmadı")
                self.confidence_label.configure(text="Güven: -")
                self.progress_var.set(0)
                
            except Exception as e:
                messagebox.showerror("Hata", f"ZIP görüntüsü yüklenirken hata oluştu:\n{str(e)}")
    
    def display_zip_image(self, image_path_in_zip):
        """ZIP içindeki görüntüyü gösterir"""
        with zipfile.ZipFile(self.current_zip_file, 'r') as zip_file:
            with zip_file.open(image_path_in_zip) as image_file:
                image_data = image_file.read()
                image = Image.open(io.BytesIO(image_data))
                
                # Görüntü boyutunu hesapla (en fazla 400x400)
                max_size = 400
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Tkinter için dönüştür
                photo = ImageTk.PhotoImage(image)
                
                # Görüntüyü göster
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Referansı koru
    
    def display_image(self, image_path):
        """Seçilen görüntüyü arayüzde gösterir"""
        try:
            # Görüntüyü yükle ve yeniden boyutlandır
            image = Image.open(image_path)
            
            # Görüntü boyutunu hesapla (en fazla 400x400)
            max_size = 400
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Tkinter için dönüştür
            photo = ImageTk.PhotoImage(image)
            
            # Görüntüyü göster
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Referansı koru
            
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü yüklenirken hata oluştu:\n{str(e)}")
    
    def predict_image(self):
        """Seçilen görüntü üzerinde tahmin yapar"""
        if (not self.current_image_path and not self.current_zip_image) or not self.model:
            messagebox.showwarning("Uyarı", "Lütfen önce bir görüntü seçin ve modelin yüklendiğinden emin olun!")
            return
        
        # Tahmin işlemini ayrı thread'de çalıştır
        self.predict_btn.configure(state='disabled', text="🔄 Tahmin yapılıyor...")
        self.status_label.configure(text="🔄 Tahmin yapılıyor...")
        
        # İlerleme çubuğunu başlat
        self.progress_var.set(30)
        self.root.update()
        
        thread = threading.Thread(target=self._predict_worker)
        thread.daemon = True
        thread.start()
    
    def _predict_worker(self):
        """Tahmin işlemini gerçekleştirir (worker thread)"""
        try:
            # Görüntü dönüşümü
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Görüntüyü yükle
            if self.current_zip_image:
                # ZIP dosyasından görüntü yükle
                with zipfile.ZipFile(self.current_zip_file, 'r') as zip_file:
                    with zip_file.open(self.current_zip_image) as image_file:
                        image_data = image_file.read()
                        image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                # Normal dosyadan görüntü yükle
                image = Image.open(self.current_image_path).convert('RGB')
            
            # Görüntüyü dönüştür
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            self.progress_var.set(60)
            self.root.update()
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted.item()]
                confidence_score = confidence.item()
            
            self.progress_var.set(100)
            self.root.update()
            
            # Sonuçları güncelle (ana thread'de)
            self.root.after(0, self._update_prediction_results, predicted_class, confidence_score, probabilities)
            
        except Exception as e:
            self.root.after(0, self._handle_prediction_error, str(e))
    
    def _update_prediction_results(self, predicted_class, confidence_score, probabilities):
        """Tahmin sonuçlarını arayüzde günceller"""
        # Hızlı özet güncelle
        self.top_prediction_label.configure(text=f"🎯 En Yüksek: {predicted_class.upper()}")
        self.confidence_label.configure(text=f"📊 Güven: %{confidence_score*100:.2f}")
        
        # Detaylı sonuçları güncelle
        self.update_predictions_display(probabilities)
        
        # Buton ve durum çubuğunu güncelle
        self.predict_btn.configure(state='normal', text="🔮 Tahmin Yap")
        self.status_label.configure(text=f"✅ Analiz tamamlandı: {predicted_class} (%{confidence_score*100:.1f})")
    
    def _handle_prediction_error(self, error_message):
        """Tahmin hatalarını işler"""
        self.predict_btn.configure(state='normal', text="🔮 Tahmin Yap")
        self.status_label.configure(text="❌ Tahmin sırasında hata oluştu")
        self.progress_var.set(0)
        messagebox.showerror("Tahmin Hatası", f"Tahmin yapılırken hata oluştu:\n{error_message}")
    
    def batch_test_images(self):
        """ZIP içindeki tüm görüntüleri test eder"""
        if not self.zip_images or not self.model:
            messagebox.showwarning("Uyarı", "ZIP dosyası yüklü değil veya model hazır değil!")
            return
        
        # Onay dialog'u
        result = messagebox.askyesno(
            "Toplu Test", 
            f"ZIP içindeki {len(self.zip_images)} görüntünün tamamını test etmek istiyor musunuz?\n\nBu işlem biraz zaman alabilir."
        )
        
        if not result:
            return
        
        # Toplu test başlat
        self.is_batch_testing = True
        self.batch_results = {}
        
        # Butonları devre dışı bırak
        self.batch_test_btn.configure(state='disabled', text="🔄 Toplu test yapılıyor...")
        self.predict_btn.configure(state='disabled')
        self.select_image_btn.configure(state='disabled')
        self.select_zip_btn.configure(state='disabled')
        
        # İlerleme sıfırla
        self.progress_var.set(0)
        self.status_label.configure(text="🚀 Toplu test başlatılıyor...")
        
        # Thread'de çalıştır
        thread = threading.Thread(target=self._batch_test_worker)
        thread.daemon = True
        thread.start()
    
    def _batch_test_worker(self):
        """Toplu test işlemini gerçekleştirir"""
        try:
            total_images = len(self.zip_images)
            
            # Görüntü dönüşümü
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            for i, image_path in enumerate(self.zip_images):
                try:
                    # İlerleme güncelle
                    progress = (i / total_images) * 100
                    self.progress_var.set(progress)
                    
                    image_name = os.path.basename(image_path)
                    self.root.after(0, lambda name=image_name: self.status_label.configure(
                        text=f"🔄 Test ediliyor: {name} ({i+1}/{total_images})"
                    ))
                    
                    # ZIP'ten görüntü yükle
                    with zipfile.ZipFile(self.current_zip_file, 'r') as zip_file:
                        with zip_file.open(image_path) as image_file:
                            image_data = image_file.read()
                            image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    
                    # Görüntüyü dönüştür ve tahmin yap
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        predicted_class = self.class_names[predicted.item()]
                        confidence_score = confidence.item()
                    
                    # Sonucu kaydet
                    self.batch_results[image_path] = {
                        'predicted_class': predicted_class,
                        'confidence': confidence_score,
                        'probabilities': probabilities[0].cpu().numpy()
                    }
                    
                except Exception as e:
                    print(f"Hata - {image_path}: {str(e)}")
                    self.batch_results[image_path] = {
                        'error': str(e)
                    }
            
            # Tamamlandı
            self.progress_var.set(100)
            self.root.after(0, self._batch_test_completed)
            
        except Exception as e:
            self.root.after(0, self._batch_test_error, str(e))
    
    def _batch_test_completed(self):
        """Toplu test tamamlandığında çalışır"""
        self.is_batch_testing = False
        
        # Butonları tekrar etkinleştir
        self.batch_test_btn.configure(state='normal', text="🚀 Tüm Görüntüleri Test Et")
        self.predict_btn.configure(state='normal')
        self.select_image_btn.configure(state='normal')
        self.select_zip_btn.configure(state='normal')
        
        # Sonuçları göster
        self.show_batch_results()
        
        self.status_label.configure(text="✅ Toplu test tamamlandı!")
    
    def _batch_test_error(self, error_message):
        """Toplu test hatası"""
        self.is_batch_testing = False
        
        # Butonları tekrar etkinleştir
        self.batch_test_btn.configure(state='normal', text="🚀 Tüm Görüntüleri Test Et")
        self.predict_btn.configure(state='normal')
        self.select_image_btn.configure(state='normal')
        self.select_zip_btn.configure(state='normal')
        
        self.status_label.configure(text="❌ Toplu test sırasında hata oluştu")
        messagebox.showerror("Toplu Test Hatası", f"Toplu test sırasında hata oluştu:\n{error_message}")
    
    def show_batch_results(self):
        """Toplu test sonuçlarını gösterir"""
        if not self.batch_results:
            return
        
        # Yeni pencere oluştur
        results_window = tk.Toplevel(self.root)
        results_window.title("📊 Toplu Test Sonuçları")
        results_window.geometry("800x600")
        results_window.configure(bg=self.colors['bg'])
        
        # Başlık
        title_label = tk.Label(
            results_window,
            text="📊 Toplu Test Sonuçları",
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Segoe UI', 16, 'bold')
        )
        title_label.pack(pady=20)
        
        # Özet bilgiler
        summary_frame = tk.Frame(results_window, bg=self.colors['card'])
        summary_frame.pack(fill='x', padx=20, pady=10)
        
        total_images = len(self.batch_results)
        successful_tests = len([r for r in self.batch_results.values() if 'error' not in r])
        failed_tests = total_images - successful_tests
        
        summary_text = f"📈 Toplam: {total_images} | ✅ Başarılı: {successful_tests} | ❌ Hatalı: {failed_tests}"
        
        tk.Label(
            summary_frame,
            text=summary_text,
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=10)
        
        # Sonuçlar tablosu
        table_frame = tk.Frame(results_window, bg=self.colors['bg'])
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Treeview
        columns = ('Görüntü', 'Tahmin', 'Güven', 'Durum')
        results_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='headings',
            style='Modern.Treeview'
        )
        
        # Sütun başlıkları
        results_tree.heading('Görüntü', text='🖼️ Görüntü')
        results_tree.heading('Tahmin', text='🎯 Tahmin')
        results_tree.heading('Güven', text='📊 Güven')
        results_tree.heading('Durum', text='✅ Durum')
        
        # Sütun genişlikleri
        results_tree.column('Görüntü', width=200, anchor='w')
        results_tree.column('Tahmin', width=150, anchor='w')
        results_tree.column('Güven', width=100, anchor='center')
        results_tree.column('Durum', width=100, anchor='center')
        
        # Scrollbar
        results_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=results_tree.yview)
        results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        # Sonuçları ekle
        for image_path, result in self.batch_results.items():
            image_name = os.path.basename(image_path)
            
            if 'error' in result:
                results_tree.insert('', 'end', values=(
                    image_name,
                    "HATA",
                    "-",
                    "❌ Başarısız"
                ), tags=('error',))
            else:
                confidence_percent = f"%{result['confidence']*100:.1f}"
                results_tree.insert('', 'end', values=(
                    image_name,
                    result['predicted_class'].title(),
                    confidence_percent,
                    "✅ Başarılı"
                ), tags=('success',))
        
        # Tag renklerini ayarla
        results_tree.tag_configure('success', background='#d4edda', foreground='#155724')
        results_tree.tag_configure('error', background='#f8d7da', foreground='#721c24')
        
        # Pack
        results_tree.pack(side='left', fill='both', expand=True)
        results_scrollbar.pack(side='right', fill='y')
        
        # Kapatma butonu
        close_btn = tk.Button(
            results_window,
            text="❌ Kapat",
            command=results_window.destroy,
            bg=self.colors['error'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            cursor='hand2'
        )
        close_btn.pack(pady=20)
    
    def run(self):
        """Uygulamayı başlatır"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleModelTestGUI()
    app.run() 