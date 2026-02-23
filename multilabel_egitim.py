# -*- coding: utf-8 -*-
"""
MULTILABEL SINIFLANDIRMA EĞİTİM SCRIPTİ
Bu script, MultiLabel_Dataset klasöründeki verileri kullanarak
bir kumaşta birden fazla hatayı aynı anda tespit edebilen model eğitir.
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import random

# ======================== AYARLAR ========================
# Bu kısmı kendi sistemine göre değiştirebilirsin
VERI_SETI_KLASORU = 'MultiLabel_Dataset'  # Veri setinin ana klasörü
CSV_DOSYASI = os.path.join(VERI_SETI_KLASORU, 'veri_etiketleri.csv')  # Etiket CSV'si
RESIM_KLASORU = os.path.join(VERI_SETI_KLASORU, 'images')  # Resimlerin olduğu klasör

IMG_BOYUT = (512, 512)  # Resim boyutu (yükseklik, genişlik)
BATCH_SIZE = 8  # Batch boyutu (GPU belleğine göre 8-16 arası idealdir)
EPOCH = 50  # Maksimum epoch sayısı (early stop durduracak)
LEARNING_RATE = 1e-4  # İnce ayar için learning rate (önce 1e-3, sonra 1e-4)
TEST_ORANI = 0.15  # Doğrulama için ayrılacak oran (test seti zaten ayrılmıştı, ama biz train'den bir kısmını validasyon olarak kullanacağız)
RANDOM_SEED = 42  # Tekrarlanabilirlik için

# ======================== VERİYİ YÜKLE ========================
print("📁 Veri seti yükleniyor...")
df = pd.read_csv(CSV_DOSYASI)
print(f"Toplam {len(df)} görüntü bulundu.")

# CSV'de 'set' sütunu var mı kontrol et (biz oluşturduk)
if 'set' in df.columns:
    train_df = df[df['set'] == 'train'].copy()
    test_df = df[df['set'] == 'test'].copy()
    print(f"Eğitim seti: {len(train_df)} görüntü (bunların içinde hibritler de var)")
    print(f"Test seti  : {len(test_df)} görüntü (sadece orijinal, asla eğitimde kullanılmayacak)")
else:
    # Eğer set sütunu yoksa, elle böl (ama seninkinde var)
    print("Uyarı: 'set' sütunu bulunamadı, rastgele bölünecek.")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

# Eğitim setini tekrar eğitim ve doğrulama olarak böl
train_df, val_df = train_test_split(
    train_df,
    test_size=TEST_ORANI,
    random_state=RANDOM_SEED,
    stratify=None  # çok etiketli olduğu için stratify kullanmak zor, rastgele bölüyoruz
)
print(f"Eğitim  (train)  : {len(train_df)}")
print(f"Doğrulama (val)  : {len(val_df)}")
print(f"Test     (test)   : {len(test_df)} (saklı tutulacak, en son kullanılacak)")

# ======================== ETİKETLERİ ÇOK ETİKETLİ FORMATTA HAZIRLA ========================
# CSV'deki 'etiketler' sütunu boşlukla ayrılmış etiketleri içeriyor (örn: "hole stain")
# Bunları listeye çeviriyoruz
train_etiket_list = [str(etiket).split() for etiket in train_df['etiketler']]
val_etiket_list = [str(etiket).split() for etiket in val_df['etiketler']]
test_etiket_list = [str(etiket).split() for etiket in test_df['etiketler']]

# MultiLabelBinarizer ile tüm sınıfları belirle ve binary vektörlere çevir
mlb = MultiLabelBinarizer()
# Tüm etiketleri birleştirip fit yapalım (böylece tüm sınıfları görsün)
tum_etiketler = train_etiket_list + val_etiket_list + test_etiket_list
mlb.fit(tum_etiketler)

sinif_isimleri = mlb.classes_
num_classes = len(sinif_isimleri)
print(f"\n🔤 Sınıflar ({num_classes} adet): {sinif_isimleri}")

# Şimdi her seti dönüştür
y_train = mlb.transform(train_etiket_list)
y_val = mlb.transform(val_etiket_list)
y_test = mlb.transform(test_etiket_list)


# ======================== VERİ JENERATÖRÜ (SEQUENCE) ========================
class MultilabelGenerator(Sequence):
    """
    Keras Sequence sınıfından türeyen özel veri jeneratörü.
    - Verileri anında yükler, hafızayı şişirmez.
    - Veri artırma (augmentation) yapabilir.
    - MixUp desteği (isteğe bağlı)
    """

    def __init__(self, dataframe, etiket_list, img_dir, batch_size, img_size,
                 mlb, sinif_isimleri, augment=True, mixup_prob=0.3):
        """
        dataframe: pandas DataFrame (en az 'dosya_adi' sütunu olmalı)
        etiket_list: her satır için etiket listesi (string listeleri)
        img_dir: resimlerin bulunduğu klasör
        batch_size: batch boyutu
        img_size: (yükseklik, genişlik)
        mlb: fit edilmiş MultiLabelBinarizer
        sinif_isimleri: sınıf isimleri listesi (mlb.classes_)
        augment: veri artırma yapılsın mı?
        mixup_prob: MixUp uygulama olasılığı (0 ise kapalı)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.etiket_list = etiket_list
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.mlb = mlb
        self.sinif_isimleri = sinif_isimleri
        self.augment = augment
        self.mixup_prob = mixup_prob
        self.indices = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def on_epoch_end(self):
        # Her epoch sonunda index'leri karıştır
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        # Bir batch için index'leri al
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for i in batch_indices:
            # Resmi yükle
            row = self.dataframe.iloc[i]
            img_path = os.path.join(self.img_dir, row['dosya_adi'])
            img = self._load_image(img_path)

            # Etiketi al (zaten y_train vb. var ama biz tekrar çevirmemek için direk etiket_list kullanabiliriz)
            # Daha hızlı olması için etiket_list'ten alalım
            etiket_str_list = self.etiket_list[i]
            label = self.mlb.transform([etiket_str_list])[0]  # one-hot vektör

            # MixUp uygula (eğer olasılık dahilindeyse)
            if np.random.random() < self.mixup_prob:
                # Rastgele başka bir index seç
                j = np.random.choice(len(self.dataframe))
                img2_path = os.path.join(self.img_dir, self.dataframe.iloc[j]['dosya_adi'])
                img2 = self._load_image(img2_path)
                label2 = self.mlb.transform([self.etiket_list[j]])[0]

                # Karışım oranı (0.3-0.7 arası)
                alpha = np.random.uniform(0.3, 0.7)
                img = cv2.addWeighted(img, alpha, img2, 1 - alpha, 0)
                # Etiketleri de aynı oranda karıştır (soft etiket)
                label = alpha * label + (1 - alpha) * label2

            # Veri artırma (augmentation)
            if self.augment:
                img = self._augment(img)

            # ResNet ön işleme (normalizasyon)
            img = preprocess_input(img.astype(np.float32))

            batch_images.append(img)
            batch_labels.append(label)

        return np.array(batch_images), np.array(batch_labels)

    def _load_image(self, path):
        """Türkçe karakter sorunu olmadan resim yükle"""
        with open(path, "rb") as f:
            bytes_data = f.read()
        np_array = np.frombuffer(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Hata: {path} okunamadı, siyah resim döndürülüyor.")
            return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        # BGR'dan RGB'ye çevir (ResNet RGB bekler)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Boyutlandır (zaten kaydederken 512x512 yapmıştık ama emin olalım)
        if img.shape[:2] != self.img_size:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        return img

    def _augment(self, img):
        """Basit veri artırma teknikleri"""
        # Yatay çevirme
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        # Parlaklık ayarı
        if np.random.rand() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Küçük döndürme (max 10 derece)
        if np.random.rand() > 0.7:
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 borderMode=cv2.BORDER_REFLECT)

        # Gauss gürültüsü (hafif)
        if np.random.rand() > 0.8:
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)

        return img


# Jeneratörleri oluştur
print("\n🔄 Veri jeneratörleri hazırlanıyor...")
train_gen = MultilabelGenerator(
    dataframe=train_df,
    etiket_list=train_etiket_list,
    img_dir=RESIM_KLASORU,
    batch_size=BATCH_SIZE,
    img_size=IMG_BOYUT,
    mlb=mlb,
    sinif_isimleri=sinif_isimleri,
    augment=True,
    mixup_prob=0.3  # %30 olasılıkla MixUp uygula (modelin genellemesini artırır)
)

val_gen = MultilabelGenerator(
    dataframe=val_df,
    etiket_list=val_etiket_list,
    img_dir=RESIM_KLASORU,
    batch_size=BATCH_SIZE,
    img_size=IMG_BOYUT,
    mlb=mlb,
    sinif_isimleri=sinif_isimleri,
    augment=False,  # Doğrulama setinde artırma yok
    mixup_prob=0.0
)

# Test jeneratörü (en son test için kullanılacak)
test_gen = MultilabelGenerator(
    dataframe=test_df,
    etiket_list=test_etiket_list,
    img_dir=RESIM_KLASORU,
    batch_size=BATCH_SIZE,
    img_size=IMG_BOYUT,
    mlb=mlb,
    sinif_isimleri=sinif_isimleri,
    augment=False,
    mixup_prob=0.0
)

# ======================== MODEL MİMARİSİ ========================
print("\n🏗️ Model oluşturuluyor...")

# 1. AŞAMA: Sadece üst katmanları eğit (base model dondurulmuş)
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_BOYUT[0], IMG_BOYUT[1], 3)
)
base_model.trainable = False  # İlk aşamada dondur

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='sigmoid')  # Çoklu etiket için sigmoid
])

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-3),
    metrics=['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# ======================== CALLBACK'LER ========================
# Model kaydetme (en iyi doğrulama binary_accuracy'ine göre)
checkpoint = ModelCheckpoint(
    'en_iyi_multilabel_model.keras',
    monitor='val_binary_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# Learning rate düşürme (doğrulama kaybı 2 epoch artarsa 0.5 ile çarp)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# Erken durdurma (doğrulama kaybı 5 epoch artarsa durdur)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Eğitim geçmişini CSV'ye kaydet
csv_logger = CSVLogger('egitim_gecmisi.csv', append=True)

callbacks = [checkpoint, lr_scheduler, early_stop, csv_logger]

# ======================== 1. AŞAMA EĞİTİMİ ========================
print("\n🚀 1. Aşama: Sadece üst katmanlar eğitiliyor (base model dondurulmuş)...")
history1 = model.fit(
    train_gen,
    epochs=10,  # İlk aşamada 10 epoch yeterli
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ======================== 2. AŞAMA: TÜM KATMANLARI İNCE AYAR ========================
print("\n🔧 2. Aşama: Tüm katmanlar ince ayar yapılıyor...")
# Base modeli aç
base_model.trainable = True

# Daha düşük learning rate ile yeniden derle
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Toplam epoch sayısı - ilk aşamada 10 yaptık, kalan 40 epoch
kalan_epoch = EPOCH - 10
if kalan_epoch > 0:
    history2 = model.fit(
        train_gen,
        epochs=kalan_epoch,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=10  # Kaldığı yerden devam et
    )
else:
    print("İlk aşama zaten tamamlandı, ikinci aşama atlanıyor.")

print("\n✅ Eğitim tamamlandı! En iyi model 'en_iyi_multilabel_model.keras' olarak kaydedildi.")

# ======================== TEST SETİNDE DEĞERLENDİR ========================
print("\n🧪 Test seti değerlendiriliyor...")
# En iyi modeli yükle
best_model = tf.keras.models.load_model('en_iyi_multilabel_model.keras')

test_loss, test_acc, test_precision, test_recall = best_model.evaluate(test_gen)
print(f"\n📊 Test Sonuçları:")
print(f"   Kayıp (Loss)      : {test_loss:.4f}")
print(f"   Doğruluk (Accuracy): {test_acc:.4f}")
print(f"   Kesinlik (Precision): {test_precision:.4f}")
print(f"   Duyarlılık (Recall): {test_recall:.4f}")

# ======================== ÖRNEK TAHMİN ========================
print("\n🔍 Örnek tahminler gösteriliyor...")
# Test setinden rastgele 5 resim seç
test_indices = np.random.choice(len(test_gen), min(5, len(test_gen)), replace=False)
for idx in test_indices:
    # Batch'ten bir örnek al
    X_batch, y_batch = test_gen[idx]
    # İlk resmi seç
    X = X_batch[0:1]  # batch boyutu 1 yap
    y_true = y_batch[0]

    # Tahmin yap
    y_pred = best_model.predict(X, verbose=0)[0]

    # Tahminleri eşik değere göre sınıflandır (0.5)
    esik = 0.5
    tahmin_edilen_siniflar = [sinif_isimleri[i] for i, p in enumerate(y_pred) if p > esik]
    gercek_siniflar = [sinif_isimleri[i] for i, val in enumerate(y_true) if val > 0]

    print(f"\nResim: {test_df.iloc[idx]['dosya_adi']}")
    print(f"   Gerçek etiketler      : {gercek_siniflar}")
    print(f"   Tahmin edilen (eşik {esik}): {tahmin_edilen_siniflar}")
    print(f"   Olasılıklar           : {', '.join([f'{sinif_isimleri[i]}: {p:.3f}' for i, p in enumerate(y_pred)])}")

# ======================== EĞİTİM GRAFİKLERİ ========================
print("\n📈 Eğitim grafikleri çiziliyor...")


def plot_training(history1, history2=None):
    plt.figure(figsize=(12, 5))

    # Kayıp grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['loss'], label='Train Loss (1. aşama)')
    plt.plot(history1.history['val_loss'], label='Val Loss (1. aşama)')
    if history2:
        plt.plot(range(10, 10 + len(history2.history['loss'])), history2.history['loss'], label='Train Loss (2. aşama)')
        plt.plot(range(10, 10 + len(history2.history['val_loss'])), history2.history['val_loss'],
                 label='Val Loss (2. aşama)')
    plt.title('Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Doğruluk grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['binary_accuracy'], label='Train Acc (1. aşama)')
    plt.plot(history1.history['val_binary_accuracy'], label='Val Acc (1. aşama)')
    if history2:
        plt.plot(range(10, 10 + len(history2.history['binary_accuracy'])), history2.history['binary_accuracy'],
                 label='Train Acc (2. aşama)')
        plt.plot(range(10, 10 + len(history2.history['val_binary_accuracy'])), history2.history['val_binary_accuracy'],
                 label='Val Acc (2. aşama)')
    plt.title('Model Doğruluğu (Binary Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('egitim_grafikleri.png', dpi=150)
    plt.show()


# Eğer history2 tanımlıysa onu da kullan, yoksa sadece history1
if 'history2' in locals():
    plot_training(history1, history2)
else:
    plot_training(history1, None)

print("✅ Grafikler 'egitim_grafikleri.png' olarak kaydedildi.")