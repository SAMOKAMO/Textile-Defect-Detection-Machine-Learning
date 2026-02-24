# -*- coding: utf-8 -*-
"""
Merkezi yapılandırma dosyası.
Tüm sabitler burada tanımlanır; diğer scriptler buradan import eder.
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# ---------- Yollar ----------
MODEL_YOLU    = str(BASE_DIR / 'models' / 'best_model.keras')
VERI_KLASORU  = str(BASE_DIR / 'data' / 'MultiLabel_Dataset')
RESIM_KLASORU = str(BASE_DIR / 'data' / 'MultiLabel_Dataset' / 'images')
CSV_YOLU      = str(BASE_DIR / 'data' / 'MultiLabel_Dataset' / 'veri_etiketleri.csv')

# ---------- Model ----------
IMG_BOYUT   = (512, 512)  # Modelin beklediği giriş boyutu
ESIK        = 0.5         # Genel tahmin eşiği
ESIK_CANLI  = 0.6         # Canlı demo eşiği (yanlış pozitifi azaltmak için)

# ---------- Eğitim ----------
BATCH_SIZE   = 8
EPOCH        = 50
EPOCH_ASAMA1 = 10    # 1. aşama epoch sayısı (sadece üst katmanlar)
LR_ASAMA1    = 1e-3  # 1. aşama learning rate
LR_ASAMA2    = 1e-4  # 2. aşama learning rate (ince ayar)
TEST_ORANI   = 0.15  # train içinden validation'a ayrılan oran
RANDOM_SEED  = 42
MIXUP_PROB   = 0.3

# ---------- Veri Hazırlama (prepare_dataset.py) ----------
HIBRIT_SAYISI      = 300   # üretilecek sentetik hibrit görsel sayısı
DATASET_TEST_ORANI = 0.2   # ham veriden test setine ayrılan oran
