# -*- coding: utf-8 -*-
"""
Kendi Fotoğrafımı Test Et
Bu script, senin belirleyeceğin bir fotoğrafı alır,
model ile tahmin yapar ve sonucu gösterir.
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import IMG_BOYUT, ESIK, MODEL_YOLU, CSV_YOLU
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import MultiLabelBinarizer

# -------------------- AYARLAR --------------------
# ★★★ BURAYA KENDİ FOTOĞRAFININ TAM DOSYA YOLUNU YAZ ★★★
FOTOGRAF_YOLU = r"DENEME STAIN.jpeg"  # Örnek yol, sen değiştir!

# -------------------- YOL DOĞRULAMA --------------------
if not Path(CSV_YOLU).exists():
    raise FileNotFoundError(
        f"CSV dosyası bulunamadı: {CSV_YOLU}\n"
        "'prepare_dataset.py' scriptini önce çalıştırın."
    )
if not Path(MODEL_YOLU).exists():
    raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_YOLU}")

# -------------------- SINIF LİSTESİNİ CSV'DEN YÜKLE --------------------
df = pd.read_csv(CSV_YOLU)

tum_etiketler = set()
for etiket_str in df['etiketler'].dropna():
    for etiket in etiket_str.split():
        tum_etiketler.add(etiket.strip())
tum_siniflar = sorted(tum_etiketler)
print(f"🔠 Sınıflar: {tum_siniflar}")

# MultiLabelBinarizer (sınıf isimlerini bilmesi için)
mlb = MultiLabelBinarizer(classes=tum_siniflar)
mlb.fit([tum_siniflar])

# -------------------- MODELİ YÜKLE --------------------
print("🧠 Model yükleniyor...")
try:
    model = load_model(MODEL_YOLU)
    print(f"✅ Model başarıyla yüklendi: {MODEL_YOLU}")
except Exception as e:
    raise RuntimeError(f"Model yüklenemedi: {MODEL_YOLU}\nHata: {e}")


# -------------------- RESİM YÜKLEME FONKSİYONU --------------------
def resim_yukle(resim_yolu):
    """Resmi diskten oku, RGB'ye çevir, yeniden boyutlandır ve ön işleme uygula."""
    with open(resim_yolu, "rb") as f:
        bytes_data = f.read()
    img_array = np.frombuffer(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"❌ Resim okunamadı: {resim_yolu}")
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


# -------------------- ANA KISIM --------------------
if __name__ == "__main__":
    # Dosyanın var olduğunu kontrol et
    if not os.path.exists(FOTOGRAF_YOLU):
        print(f"❌ Dosya bulunamadı: {FOTOGRAF_YOLU}")
        print("Lütfen FOTOGRAF_YOLU değişkenini doğru şekilde düzenle.")
        exit()

    print(f"\n📸 Test edilen fotoğraf: {FOTOGRAF_YOLU}")

    # Tahmin yap
    aktif_siniflar, aktif_olasiliklar, tum_olasiliklar = tahmin_et(FOTOGRAF_YOLU)

    # Sonuçları göster
    print("\n🔮 Model Tahmini:")
    if aktif_siniflar:
        print("✅ Tespit edilen hatalar:")
        for sinif, olasilik in zip(aktif_siniflar, aktif_olasiliklar):
            print(f"   - {sinif}: %{olasilik * 100:.2f}")
    else:
        print("❌ Hiçbir hata tespit edilmedi (defect_free olabilir).")

    # Tüm sınıf olasılıklarını göster (isteğe bağlı)
    print("\n📊 Tüm sınıf olasılıkları:")
    for sinif, olasilik in zip(tum_siniflar, tum_olasiliklar):
        print(f"   {sinif}: {olasilik:.4f}")

    print("\n✅ İşlem tamamlandı.")