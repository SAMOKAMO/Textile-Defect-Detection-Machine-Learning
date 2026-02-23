# -*- coding: utf-8 -*-
"""
Rastgele Mix (Hibrit) Fotoğrafları Test Etme
Bu script, MultiLabel_Dataset içindeki sadece mix_ ile başlayan resimlerden rastgele birini seçer,
model ile tahmin yapar ve sonucu gösterir.
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
MODEL_YOLU = 'en_iyi_multilabel_model.keras'  # Kullanılacak model (yoksa son modeli dener)
IMG_BOYUT = (512, 512)  # Modelin beklediği boyut
ESIK = 0.5  # Eşik değeri (0.5 üstü pozitif)

# -------------------- SINIF LİSTESİNİ CSV'DEN YÜKLE --------------------
csv_path = os.path.join(VERI_KLASORU, 'veri_etiketleri.csv')
df = pd.read_csv(csv_path)

tum_etiketler = set()
for etiket_str in df['etiketler'].dropna():
    for etiket in etiket_str.split():
        tum_etiketler.add(etiket.strip())
tum_siniflar = sorted(tum_etiketler)
print(f" Sınıflar: {tum_siniflar}")

# MultiLabelBinarizer'ı hazırla (sınıf isimlerini bilmesi için)
mlb = MultiLabelBinarizer(classes=tum_siniflar)
mlb.fit([tum_siniflar])

# -------------------- MIX FOTOĞRAFLARIN LİSTESİNİ AL --------------------
resim_klasoru = os.path.join(VERI_KLASORU, 'images')
tum_dosyalar = os.listdir(resim_klasoru)
mix_dosyalar = [f for f in tum_dosyalar if f.startswith('mix_') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f" Toplam {len(mix_dosyalar)} mix resim bulundu.")

if len(mix_dosyalar) == 0:
    print(" Hiç mix resim yok! Program sonlandırılıyor.")
    exit()

# -------------------- MODELİ YÜKLE --------------------
print(" Model yükleniyor...")
try:
    model = load_model(MODEL_YOLU)
    print(f" Model {MODEL_YOLU} başarıyla yüklendi.")
except:
    MODEL_YOLU = 'multilabel_model_son.keras'
    model = load_model(MODEL_YOLU)
    print(f" Model {MODEL_YOLU} başarıyla yüklendi.")


# -------------------- RESİM YÜKLEME FONKSİYONU --------------------
def resim_yukle(resim_yolu):
    """Resmi diskten oku, RGB'ye çevir, yeniden boyutlandır ve ön işleme uygula."""
    with open(resim_yolu, "rb") as f:
        bytes_data = f.read()
    img_array = np.frombuffer(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f" Resim okunamadı: {resim_yolu}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_BOYUT[1], IMG_BOYUT[0]))
    img = preprocess_input(img.astype(np.float32))
    return img


# -------------------- GERÇEK ETİKETİ BUL --------------------
def gercek_etiketi_bul(dosya_adi):
    """CSV'den verilen dosya adının gerçek etiketlerini döndürür."""
    satir = df[df['dosya_adi'] == dosya_adi]
    if len(satir) == 0:
        return None
    etiket_str = satir.iloc[0]['etiketler']
    if pd.isna(etiket_str) or etiket_str == '':
        return []
    return [e.strip() for e in etiket_str.split()]


# -------------------- TAHMİN YAP --------------------
def tahmin_et(resim_yolu):
    img = resim_yukle(resim_yolu)
    img_batch = np.expand_dims(img, axis=0)
    tahminler = model.predict(img_batch, verbose=0)[0]
    aktif_indeksler = np.where(tahminler >= ESIK)[0]
    aktif_siniflar = [tum_siniflar[i] for i in aktif_indeksler]
    aktif_olasiliklar = tahminler[aktif_indeksler]
    return aktif_siniflar, aktif_olasiliklar, tahminler


# -------------------- ANA DÖNGÜ --------------------
if __name__ == "__main__":
    print("\n Rastgele bir mix resim seçiliyor...\n")

    # Rastgele bir mix dosya seç
    secilen_dosya = random.choice(mix_dosyalar)
    resim_yolu = os.path.join(resim_klasoru, secilen_dosya)

    print(f" Seçilen resim: {secilen_dosya}")

    # Gerçek etiket
    gercek_etiketler = gercek_etiketi_bul(secilen_dosya)
    print(f"Gerçek etiket(ler): {gercek_etiketler if gercek_etiketler else 'Etiket yok'}")

    # Tahmin
    aktif_siniflar, aktif_olasiliklar, tum_olasiliklar = tahmin_et(resim_yolu)

    print("\n Model Tahmini:")
    if aktif_siniflar:
        print(" Tespit edilen hatalar:")
        for sinif, olasilik in zip(aktif_siniflar, aktif_olasiliklar):
            print(f"   - {sinif}: %{olasilik * 100:.2f}")
    else:
        print(" Hiçbir hata tespit edilmedi.")

    # Tüm sınıf olasılıkları
    print("\n Tüm sınıf olasılıkları:")
    for sinif, olasilik in zip(tum_siniflar, tum_olasiliklar):
        print(f"   {sinif}: {olasilik:.4f}")

    # Gerçek ile tahmini karşılaştır
    print("\n Karşılaştırma:")
    gercek_kume = set(gercek_etiketler) if gercek_etiketler else set()
    tahmin_kume = set(aktif_siniflar)

    dogru_tahminler = gercek_kume & tahmin_kume
    eksik_tahminler = gercek_kume - tahmin_kume
    fazla_tahminler = tahmin_kume - gercek_kume

    if dogru_tahminler:
        print(f" Doğru tespit edilenler: {dogru_tahminler}")
    if eksik_tahminler:
        print(f" Kaçırılanlar (yanlış negatif): {eksik_tahminler}")
    if fazla_tahminler:
        print(f"️ Fazladan tespit edilenler (yanlış pozitif): {fazla_tahminler}")

    if not eksik_tahminler and not fazla_tahminler:
        print(" Mükemmel! Tüm etiketler doğru tespit edildi.")

    print("\n Test tamamlandı.")