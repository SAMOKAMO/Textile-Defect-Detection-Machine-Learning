import cv2
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import IMG_BOYUT, ESIK_CANLI, MODEL_YOLU, CSV_YOLU
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 1. AYARLAR VE MODEL YÜKLEME ---
ESIK_DEGERI = ESIK_CANLI  # Yanlış pozitifi azaltmak için yüksek eşik

# -------------------- YOL DOĞRULAMA --------------------
if not Path(MODEL_YOLU).exists():
    raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_YOLU}")

# Sınıfları CSV'den oku; CSV yoksa eğitimde kullanılan bilinen sınıflarla devam et
try:
    df = pd.read_csv(CSV_YOLU)
    tum_etiketler = set()
    for etiket_str in df['etiketler'].dropna():
        for etiket in str(etiket_str).split():
            tum_etiketler.add(etiket.strip())
    SINIFLAR = sorted(tum_etiketler)
    print(f"Sınıflar CSV'den yüklendi: {SINIFLAR}")
except FileNotFoundError:
    SINIFLAR = ['defect_free', 'hole', 'horizontal', 'lines', 'stain', 'vertical']
    print(f"UYARI: CSV bulunamadı. Fallback sınıflar kullanılıyor: {SINIFLAR}")

print("Model yükleniyor, lütfen bekleyin...")
model = load_model(MODEL_YOLU)
print("✅ Model yüklendi! Kamera açılıyor...")

# --- 2. KAMERAYI AÇ ---
cap = cv2.VideoCapture(0)

# Kamera çözünürlüğünü yüksek ayarla (varsa)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_sayaci = 0
son_tahmin_metni = "Analiz Bekleniyor..."
renk = (0, 255, 0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_sayaci += 1
    h, w, _ = frame.shape

    # --- 3. HEDEF KUTU (ROI) OLUŞTUR ---
    # Ekranın tam ortasına 400x400 piksellik bir kare çizelim
    kutu_boyutu = 400
    x1 = int((w - kutu_boyutu) / 2)
    y1 = int((h - kutu_boyutu) / 2)
    x2 = x1 + kutu_boyutu
    y2 = y1 + kutu_boyutu

    # SADECE KUTUNUN İÇİNİ KES AL (Arka planı çöpe at)
    kumas_kismi = frame[y1:y2, x1:x2]

    # --- 4. ANLIK ANALİZ (Her 5 karede 1) ---
    if frame_sayaci % 5 == 0:
        # Sadece kestiğimiz kareyi modele yolla
        img_resized = cv2.resize(kumas_kismi, IMG_BOYUT)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        x = np.expand_dims(img_rgb, axis=0)
        x = preprocess_input(x.astype(np.float32))

        tahminler = model.predict(x, verbose=0)[0]

        bulunan_hatalar = []
        for i, olasilik in enumerate(tahminler):
            if olasilik > ESIK_DEGERI:
                bulunan_hatalar.append(f"{SINIFLAR[i]}")

        # Ekrana Yazılacak Karar Mekanizması
        if len(bulunan_hatalar) == 0:
            son_tahmin_metni = "HATA BULUNAMADI (Emin Degil)"
            renk = (0, 255, 255)  # Sarı
        elif "defect_free" in bulunan_hatalar and len(bulunan_hatalar) == 1:
            son_tahmin_metni = "KUMAS SAGLAM"
            renk = (0, 255, 0)  # Yeşil
        else:
            hatalar = [h for h in bulunan_hatalar if h != "defect_free"]
            if hatalar:
                son_tahmin_metni = "HATA: " + " + ".join(hatalar)
                renk = (0, 0, 255)  # Kırmızı

    # --- 5. EKRAN ÇİZİMLERİ ---
    # 1. Ortadaki Hedef Kutuyu Çiz (Kullanıcı kumaşı buraya tutsun)
    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 3)
    cv2.putText(frame, "KUMASI BU KUTUYA TUTUN", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 2. Sonucu Ekrana Yaz
    cv2.rectangle(frame, (10, 10), (700, 60), (0, 0, 0), -1)
    cv2.putText(frame, son_tahmin_metni, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 3)

    cv2.imshow('Üretim Hattı Canlı Analiz', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()