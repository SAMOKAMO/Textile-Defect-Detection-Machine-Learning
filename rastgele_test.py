# -*- coding: utf-8 -*-
"""
Rastgele Test Scripti
MultiLabel_Dataset/images klasöründen rastgele bir resim seçer,
modelden geçirir ve sonuçları gösterir.
"""

import os
import random
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import MultiLabelBinarizer

# -------------------- AYARLAR --------------------
VERI_KLASORU = 'MultiLabel_Dataset'  # Ana veri klasörü
RESIM_KLASORU = os.path.join(VERI_KLASORU, 'images')  # Resimlerin bulunduğu klasör
MODEL_YOLU = 'en_iyi_multilabel_model.keras'  # Kullanılacak model
IMG_BOYUT = (512, 512)  # Modelin beklediği boyut
ESIK = 0.5  # Eşik değeri

# -------------------- SINIF LİSTESİNİ CSV'DEN YÜKLE --------------------
csv_path = os.path.join(VERI_KLASORU, 'veri_etiketleri.csv')
df = pd.read_csv(csv_path)

tum_etiketler = set()
for etiket_str in df['etiketler'].dropna():
    for etiket in etiket_str.split():
        tum_etiketler.add(etiket.strip())
tum_siniflar = sorted(tum_etiketler)
print(f"Sınıflar: {tum_siniflar}")

# MultiLabelBinarizer'ı hazırla (sınıf isimlerini bilmesi için)
mlb = MultiLabelBinarizer(classes=tum_siniflar)
mlb.fit([tum_siniflar])

# -------------------- MODELİ YÜKLE --------------------
print("Model yükleniyor...")
try:
    model = load_model(MODEL_YOLU)
    print(f"Model {MODEL_YOLU} başarıyla yüklendi.")
except:
    MODEL_YOLU = 'multilabel_model_son.keras'
    model = load_model(MODEL_YOLU)
    print(f"Model {MODEL_YOLU} başarıyla yüklendi.")


# -------------------- RESİM YÜKLEME FONKSİYONU --------------------
def resim_yukle(resim_yolu):
    """Resmi diskten oku, RGB'ye çevir, yeniden boyutlandır ve ön işleme uygula."""
    with open(resim_yolu, "rb") as f:
        bytes_data = f.read()
    img_array = np.frombuffer(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Resim okunamadı: {resim_yolu}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_BOYUT[1], IMG_BOYUT[0]))
    img = preprocess_input(img.astype(np.float32))
    return img


# -------------------- TAHMİN YAP --------------------
def tahmin_et(resim_yolu):
    img = resim_yukle(resim_yolu)
    img_batch = np.expand_dims(img, axis=0)
    tahminler = model.predict(img_batch, verbose=0)[0]

    aktif_indeksler = np.where(tahminler >= ESIK)[0]
    aktif_siniflar = [tum_siniflar[i] for i in aktif_indeksler]
    aktif_olasiliklar = tahminler[aktif_indeksler]

    return aktif_siniflar, aktif_olasiliklar, tahminler


# -------------------- RASTGELE RESİM SEÇ --------------------
def rastgele_resim_sec():
    tum_resimler = [f for f in os.listdir(RESIM_KLASORU)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not tum_resimler:
        print("❌ Resim klasöründe hiç resim bulunamadı!")
        return None
    secilen = random.choice(tum_resimler)
    return os.path.join(RESIM_KLASORU, secilen), secilen


# -------------------- ANA DÖNGÜ --------------------
if __name__ == "__main__":
    print("\n🎲 Rastgele Test Başlıyor...")
    print("---------------------------")

    # Rastgele bir resim seç
    resim_yolu, resim_adi = rastgele_resim_sec()
    if resim_yolu is None:
        exit()

    print(f"Seçilen resim: {resim_adi}")

    # Tahmin yap
    print("🔮 Tahmin yapılıyor...")
    aktif_siniflar, aktif_olasiliklar, tum_olasiliklar = tahmin_et(resim_yolu)

    # Sonuçları göster
    print("\n📌 Sonuçlar:")
    if aktif_siniflar:
        print("✅ Tespit edilen hatalar:")
        for sinif, olasilik in zip(aktif_siniflar, aktif_olasiliklar):
            print(f"   - {sinif}: %{olasilik * 100:.2f} olasılık")
    else:
        print("❌ Hiçbir hata tespit edilmedi (defect_free olabilir).")

    # Opsiyonel: Tüm sınıf olasılıklarını göster
    print("\n📊 Tüm sınıf olasılıkları:")
    for sinif, olasilik in zip(tum_siniflar, tum_olasiliklar):
        print(f"   {sinif}: {olasilik:.4f}")

    print("\n✅ Test tamamlandı.")