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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTk
from matplotlib.figure import Figure

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
        
        # Sekme 3: Grafik
        self.chart_frame = tk.Frame(self.notebook, bg=self.colors['card'])
        self.notebook.add(self.chart_frame, text="📈 Grafik")
        
        # En yüksek 10 tahmin listesi
        self.create_top_predictions_list()
        
        # Tüm sonuçlar listesi
        self.create_all_predictions_list()
        
        # Grafik alanı
        self.create_chart_area()
        
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
        columns = ('Hayvan', 'Olasılık', 'Yüzde', 'Kategori')
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
        self.all_tree.heading('Kategori', text='🏷️ Kategori')
        
        # Sütun genişlikleri
        self.all_tree.column('Hayvan', width=120, anchor='w')
        self.all_tree.column('Olasılık', width=80, anchor='center')
        self.all_tree.column('Yüzde', width=80, anchor='center')
        self.all_tree.column('Kategori', width=100, anchor='w')
        
        # Scrollbar
        all_scrollbar = ttk.Scrollbar(self.all_frame, orient='vertical', command=self.all_tree.yview)
        self.all_tree.configure(yscrollcommand=all_scrollbar.set)
        
        # Pack
        self.all_tree.pack(side='left', fill='both', expand=True, padx=(20, 0), pady=10)
        all_scrollbar.pack(side='right', fill='y', padx=(0, 20), pady=10)
        
    def create_chart_area(self):
        """Grafik alanını oluşturur"""
        
        # Matplotlib figürü
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor='#3c3c3c')
        self.ax = self.fig.add_subplot(111, facecolor='#3c3c3c')
        
        # Canvas
        self.canvas = FigureCanvasTk(self.fig, self.chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
        
        # İlk grafik
        self.ax.text(0.5, 0.5, 'Henüz tahmin yapılmadı\n\nBir görüntü seçin ve tahmin yapın', 
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=12, color='white')
        self.ax.set_facecolor('#3c3c3c')
        self.canvas.draw()
        
    def get_animal_category(self, animal_name):
        """Hayvan kategorisini döndürür"""
        mammals = ['antelope', 'badger', 'bat', 'bear', 'bison', 'boar', 'cat', 'chimpanzee', 'cow', 'coyote', 'deer', 'dog', 'dolphin', 'donkey', 'elephant', 'fox', 'goat', 'gorilla', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'horse', 'hyena', 'kangaroo', 'koala', 'leopard', 'lion', 'mouse', 'okapi', 'orangutan', 'otter', 'ox', 'panda', 'pig', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'seal', 'sheep', 'squirrel', 'tiger', 'whale', 'wolf', 'wombat', 'zebra']
        birds = ['crow', 'duck', 'eagle', 'flamingo', 'goose', 'hornbill', 'hummingbird', 'owl', 'parrot', 'pelecaniformes', 'penguin', 'pigeon', 'sandpiper', 'sparrow', 'swan', 'turkey', 'woodpecker']
        insects = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'dragonfly', 'fly', 'grasshopper', 'ladybugs', 'mosquito', 'moth']
        sea_creatures = ['goldfish', 'jellyfish', 'lobster', 'octopus', 'oyster', 'seahorse', 'shark', 'squid', 'starfish']
        reptiles = ['lizard', 'snake', 'turtle']
        amphibians = ['crab']
        
        if animal_name in mammals:
            return '🐾 Memeli'
        elif animal_name in birds:
            return '🐦 Kuş'
        elif animal_name in insects:
            return '🐛 Böcek'
        elif animal_name in sea_creatures:
            return '🐠 Deniz'
        elif animal_name in reptiles:
            return '🐢 Sürüngen'
        elif animal_name in amphibians:
            return '🐸 Amfibi'
        else:
            return '❓ Diğer'
    
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
                category = self.get_animal_category(class_name)
                percentage = f"%{probability*100:.2f}"
                prob_str = f"{probability:.4f}"
                
                self.all_tree.insert('', 'end', values=(
                    class_name.title(), 
                    prob_str, 
                    percentage, 
                    category
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
        
        # Grafik güncelle
        self.update_chart(sorted_predictions[:15])  # En yüksek 15'i göster
    
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
            category = self.get_animal_category(class_name)
            percentage = f"%{probability*100:.2f}"
            prob_str = f"{probability:.4f}"
            
            self.all_tree.insert('', 'end', values=(
                class_name.title(), 
                prob_str, 
                percentage, 
                category
            ))
    
    def update_chart(self, top_predictions):
        """Grafik günceller"""
        
        # Grafik temizle
        self.ax.clear()
        
        # Veri hazırla
        animals = [pred[0].title() for pred in top_predictions]
        probabilities = [pred[1] * 100 for pred in top_predictions]
        
        # Renk paleti
        colors = plt.cm.Set3(np.linspace(0, 1, len(animals)))
        
        # Yatay bar grafik
        bars = self.ax.barh(range(len(animals)), probabilities, color=colors)
        
        # Grafik ayarları
        self.ax.set_yticks(range(len(animals)))
        self.ax.set_yticklabels(animals, fontsize=9, color='white')
        self.ax.set_xlabel('Olasılık (%)', fontsize=10, color='white')
        self.ax.set_title('En Yüksek 15 Tahmin', fontsize=12, color='white', pad=20)
        
        # Arka plan rengi
        self.ax.set_facecolor('#3c3c3c')
        self.fig.patch.set_facecolor('#3c3c3c')
        
        # Grid
        self.ax.grid(True, alpha=0.3, color='white')
        
        # X ekseni rengi
        self.ax.tick_params(axis='x', colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        # Değerleri bar üzerinde göster
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob > 1:  # Sadece %1'den büyük olanları göster
                self.ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.1f}%', va='center', fontsize=8, color='white')
        
        # Layout ayarla
        self.fig.tight_layout()
        
        # Grafik güncelle
        self.canvas.draw()
    
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
            self.display_image(file_path)
            self.predict_btn.configure(state='normal')
            self.status_label.configure(text="✅ Görüntü seçildi - Tahmin yapabilirsiniz")
            
            # Sonuçları sıfırla
            self.top_prediction_label.configure(text="Henüz tahmin yapılmadı")
            self.confidence_label.configure(text="Güven: -")
            self.progress_var.set(0)
    
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
        if not self.current_image_path or not self.model:
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
            
            # Görüntüyü yükle ve dönüştür
            image = Image.open(self.current_image_path).convert('RGB')
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
    
    def run(self):
        """Uygulamayı başlatır"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleModelTestGUI()
    app.run() 