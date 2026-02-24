import os
import sys
import cv2
import pandas as pd
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from itertools import combinations
sys.path.insert(0, str(Path(__file__).parent))
from config import (IMG_BOYUT, VERI_KLASORU, RESIM_KLASORU, CSV_YOLU,
                    HIBRIT_SAYISI, DATASET_TEST_ORANI)

BASE_DIR = Path(__file__).parent.parent

# --- AYARLAR ---
HEDEF_BOYUT        = IMG_BOYUT           # config.py ile tutarlı olması için
TEST_ORANI         = DATASET_TEST_ORANI  # %20 test, %80 eğitim

# Klasör yolları
TEMIZ_KLASOR       = str(BASE_DIR / 'data' / 'Temiz_Veri_Seti')
YENI_KLASOR        = VERI_KLASORU
YENI_RESIM_KLASORU = RESIM_KLASORU

# --- YARDIMCI FONKSİYONLAR (Türkçe karakter desteği) ---
def turkce_imread(dosya_yolu):
    with open(dosya_yolu, "rb") as f:
        bytes_data = f.read()
    np_array = np.frombuffer(bytes_data, dtype=np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)

def turkce_imwrite(dosya_yolu, resim):
    ret, buf = cv2.imencode(os.path.splitext(dosya_yolu)[1], resim)
    if ret:
        with open(dosya_yolu, "wb") as f:
            buf.tofile(f)

# --- 1. TÜM ORİJİNAL RESİMLERİ TARA, SINIF ADLARINI DÜZENLE ---
print("📁 Orijinal resimler taranıyor...")
orijinal_veri = []  # (yol, sinif_adi, dosya_adi)

for klasor in os.listdir(TEMIZ_KLASOR):
    klasor_yolu = os.path.join(TEMIZ_KLASOR, klasor)
    if not os.path.isdir(klasor_yolu):
        continue

    # Sınıf adını normalize et: boşluk → alt çizgi, büyük harf → küçük harf
    sinif_adi = klasor.replace(' ', '_').lower()

    for dosya in os.listdir(klasor_yolu):
        if not dosya.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        dosya_yolu = os.path.join(klasor_yolu, dosya)
        orijinal_veri.append((dosya_yolu, sinif_adi, dosya))

print(f"Toplam {len(orijinal_veri)} orijinal resim bulundu.")

# --- 2. ORİJİNAL RESİMLERİ EĞİTİM/TEST OLARAK AYIR (SINIF DAĞILIMINI KORU) ---
siniflar = list(set([item[1] for item in orijinal_veri]))
train_veri = []
test_veri = []

for sinif in siniflar:
    sinif_veri = [item for item in orijinal_veri if item[1] == sinif]
    train_idx, test_idx = train_test_split(
        range(len(sinif_veri)),
        test_size=TEST_ORANI,
        random_state=42
    )
    for i in train_idx:
        train_veri.append(sinif_veri[i])
    for i in test_idx:
        test_veri.append(sinif_veri[i])

print(f"Eğitim seti: {len(train_veri)} resim")
print(f"Test seti   : {len(test_veri)} resim")

# --- 3. YENİ KLASÖR YAPISINI HAZIRLA ---
if not os.path.exists(YENI_RESIM_KLASORU):
    os.makedirs(YENI_RESIM_KLASORU)

csv_satirlari = []

# --- 4. ORİJİNAL RESİMLERİ HEDEF BOYUTTA KAYDET ---
print("\n🖼️ Orijinal resimler yeniden boyutlandırılıp kaydediliyor...")

def orijinal_resimleri_kaydet(veri_listesi, set_adi):
    for (yol, sinif, dosya_adi) in tqdm(veri_listesi, desc=f"{set_adi} orijinaller"):
        img = turkce_imread(yol)
        if img is None:
            print(f"  Hata: {yol} okunamadı, atlanıyor.")
            continue

        # Renk kanallarını düzenle
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, HEDEF_BOYUT)

        # Yeni dosya adı: sinif_orijinalad.jpg
        yeni_isim = f"{sinif}_{dosya_adi}"
        yeni_isim = os.path.splitext(yeni_isim)[0] + ".jpg"
        yeni_yol = os.path.join(YENI_RESIM_KLASORU, yeni_isim)

        turkce_imwrite(yeni_yol, img)

        csv_satirlari.append({
            'dosya_adi': yeni_isim,
            'etiketler': sinif,
            'set': set_adi
        })

# Önce eğitim setini kaydet
orijinal_resimleri_kaydet(train_veri, 'train')
# Sonra test setini kaydet
orijinal_resimleri_kaydet(test_veri, 'test')

# --- 5. HİBRİT (KARIŞIK) RESİMLERİ ÜRET (SADECE EĞİTİM SETİNDEKİ ORİJİNALLERLE) ---
hata_siniflari = ['hole', 'horizontal', 'lines', 'stain', 'vertical']

# İkili kombinasyonları oluştur
kombinasyonlar = list(combinations(hata_siniflari, 2))
print(f"\n🔀 {len(kombinasyonlar)} farklı hata çifti var. Toplam {HIBRIT_SAYISI} hibrit üretilecek.")
print("Kombinasyonlar:", kombinasyonlar)

# Her kombinasyondan üretilecek sayı
hibrit_per_comb = HIBRIT_SAYISI // len(kombinasyonlar)
artan = HIBRIT_SAYISI % len(kombinasyonlar)

# Eğitim setindeki orijinal resimlerin sınıf bazlı listesi
train_by_class = {}
for (yol, sinif, dosya_adi) in train_veri:
    train_by_class.setdefault(sinif, []).append((yol, dosya_adi))

sayac = 0
pbar = tqdm(total=HIBRIT_SAYISI, desc="Hibrit üretiliyor")

for idx, (sinif1, sinif2) in enumerate(kombinasyonlar):
    # Bu kombinasyondan kaç tane üretilecek?
    kac_tane = hibrit_per_comb + (1 if idx < artan else 0)

    for _ in range(kac_tane):
        # Eğitim setinden rastgele birer resim seç
        yol1, dosya1 = random.choice(train_by_class[sinif1])
        yol2, dosya2 = random.choice(train_by_class[sinif2])

        img1 = turkce_imread(yol1)
        img2 = turkce_imread(yol2)

        if img1 is None or img2 is None:
            print(f"  Hata: Resim okunamadı, atlanıyor.")
            continue

        # img1'i düzenle
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        elif img1.shape[2] == 4:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2RGB)
        else:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, HEDEF_BOYUT)

        # img2'yi düzenle
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        elif img2.shape[2] == 4:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2RGB)
        else:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, HEDEF_BOYUT)

        # Karıştır
        alpha = random.uniform(0.4, 0.6)
        karisik = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

        yeni_isim = f"mix_{sayac}_{sinif1}_{sinif2}.jpg"
        yeni_yol = os.path.join(YENI_RESIM_KLASORU, yeni_isim)
        turkce_imwrite(yeni_yol, karisik)

        # CSV'ye ekle
        csv_satirlari.append({
            'dosya_adi': yeni_isim,
            'etiketler': f"{sinif1} {sinif2}",
            'set': 'train'
        })

        sayac += 1
        pbar.update(1)

pbar.close()

# --- 6. CSV'Yİ KAYDET ---
df = pd.DataFrame(csv_satirlari)
df.to_csv(CSV_YOLU, index=False, encoding='utf-8')

print(f"\n✅ Veri seti oluşturuldu!")
print(f"   Toplam resim: {len(df)}")
print(f"   Eğitim seti: {len(df[df['set']=='train'])} ({len(df[df['set']=='train'])-HIBRIT_SAYISI} orijinal + {HIBRIT_SAYISI} hibrit)")
print(f"   Test seti  : {len(df[df['set']=='test'])} (sadece orijinal)")
print(f"   CSV kaydedildi: {CSV_YOLU}")