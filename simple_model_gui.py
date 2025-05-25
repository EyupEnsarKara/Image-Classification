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

class SimpleModelTestGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 AI Model Test Arayüzü")
        self.root.geometry("1000x700")
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
        
    def create_widgets(self):
        """Modern arayüz bileşenlerini oluşturur"""
        
        # Ana başlık
        title_frame = tk.Frame(self.root, bg=self.colors['bg'], height=60)
        title_frame.pack(fill='x', padx=20, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🤖 AI Model Test Arayüzü",
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            font=('Segoe UI', 20, 'bold')
        )
        title_label.pack(expand=True)
        
        # Ana içerik çerçevesi
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Sol panel - Kontroller
        left_panel = tk.Frame(main_frame, bg=self.colors['card'], width=350)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Sol panel içeriği
        self.create_control_panel(left_panel)
        
        # Sağ panel - Görüntü
        right_panel = tk.Frame(main_frame, bg=self.colors['card'])
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Sağ panel içeriği
        self.create_image_panel(right_panel)
        
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
        
        # Sonuçlar
        result_frame = tk.Frame(parent, bg=self.colors['card'])
        result_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(
            result_frame,
            text="📈 Tahmin Sonuçları",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=(0, 15))
        
        # Tahmin sonucu
        self.prediction_label = tk.Label(
            result_frame,
            text="Henüz tahmin yapılmadı",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 10),
            wraplength=300
        )
        self.prediction_label.pack(pady=5)
        
        # Güven skoru
        self.confidence_label = tk.Label(
            result_frame,
            text="Güven: -",
            bg=self.colors['card'],
            fg=self.colors['fg'],
            font=('Segoe UI', 10)
        )
        self.confidence_label.pack(pady=5)
        
        # İlerleme çubuğu
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            result_frame,
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
            self.prediction_label.configure(text="Henüz tahmin yapılmadı")
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
            self.root.after(0, self._update_prediction_results, predicted_class, confidence_score)
            
        except Exception as e:
            self.root.after(0, self._handle_prediction_error, str(e))
    
    def _update_prediction_results(self, predicted_class, confidence_score):
        """Tahmin sonuçlarını arayüzde günceller"""
        # Sonuçları göster
        self.prediction_label.configure(text=f"🎯 Tahmin: {predicted_class.upper()}")
        self.confidence_label.configure(text=f"📊 Güven: %{confidence_score*100:.2f}")
        
        # Buton ve durum çubuğunu güncelle
        self.predict_btn.configure(state='normal', text="🔮 Tahmin Yap")
        self.status_label.configure(text=f"✅ Tahmin tamamlandı: {predicted_class} (%{confidence_score*100:.1f})")
        
        # Başarı mesajı
        messagebox.showinfo(
            "Tahmin Tamamlandı", 
            f"🎯 Tahmin: {predicted_class.upper()}\n📊 Güven Skoru: %{confidence_score*100:.2f}"
        )
    
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