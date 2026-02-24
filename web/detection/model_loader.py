"""
Singleton model loader — uygulama başlarken bir kez yüklenir,
sonra her request'te yeniden yüklenmez.
"""
import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle (src/config.py'ye ulaşmak için)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODEL_YOLU, CSV_YOLU

model = None
siniflar = None


def load():
    global model, siniflar

    import tensorflow as tf
    import pandas as pd

    print("Model yükleniyor...")
    model = tf.keras.models.load_model(MODEL_YOLU)
    print("✅ Model yüklendi.")

    try:
        df = pd.read_csv(CSV_YOLU)
        tum_etiketler = set()
        for etiket_str in df['etiketler'].dropna():
            for etiket in str(etiket_str).split():
                tum_etiketler.add(etiket.strip())
        siniflar = sorted(tum_etiketler)
        print(f"✅ Sınıflar CSV'den yüklendi: {siniflar}")
    except FileNotFoundError:
        siniflar = ['defect_free', 'hole', 'horizontal', 'lines', 'stain', 'vertical']
        print(f"⚠️  CSV bulunamadı, fallback sınıflar kullanılıyor: {siniflar}")

    model_output_size = model.output_shape[-1]
    assert len(siniflar) == model_output_size, (
        f"Class count mismatch: {len(siniflar)} class names "
        f"but model outputs {model_output_size} neurons. "
        f"Classes: {siniflar}"
    )
