# Tekstil Kumaş Hata Tespiti — Derin Öğrenme ile Kalite Kontrol

Bu proje, endüstriyel tekstil üretim hatlarında kumaş yüzeyindeki hataları gerçek zamanlı olarak tespit etmek amacıyla geliştirilmiş bir yapay zeka sistemidir. ResNet50 mimarisi üzerine transfer öğrenimi uygulanarak eğitilen model, bir kumaş parçasında aynı anda birden fazla hata türünü tespit edebilmektedir (multilabel sınıflandırma).

## Tespit Edilen Hata Türleri

| Sınıf | Açıklama |
|---|---|
| `hole` | Delik |
| `stain` | Leke |
| `lines` | Çizgi / şerit bozukluğu |
| `horizontal` | Yatay iplik hatası |
| `Vertical` | Dikey iplik hatası |
| `defect_free` | Hatasız kumaş |

## Proje Yapısı

```
├── src/
│   ├── train.py            # Model eğitimi
│   ├── live_demo.py        # Gerçek zamanlı kamera analizi
│   ├── predict.py          # Tek fotoğraf ile test
│   ├── prepare_dataset.py  # Veri seti hazırlama
│   ├── test_random.py      # Veri setinden rastgele test
│   └── test_mix.py         # Hibrit görüntülerle test
├── models/
│   └── best_model.keras    # Eğitilmiş model (Git LFS)
├── data/                   # Veri seti klasörü (repoya dahil değil)
└── requirements.txt
```

## Kurulum

### 1. Depoyu klonla

```bash
git clone https://github.com/kullanici_adi/Textile-Defect-Detection-Machine-Learning.git
cd Textile-Defect-Detection-Machine-Learning
```

### 2. Sanal ortam oluştur ve aktif et

```bash
python3 -m venv tez_env
source tez_env/bin/activate        # macOS / Linux
# tez_env\Scripts\activate         # Windows
```

### 3. Bağımlılıkları yükle

**Apple Silicon Mac (M1/M2/M3) için:**
```bash
pip install --upgrade pip
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt
```

**Intel Mac / Linux / Windows için:**
```bash
pip install --upgrade pip
pip install tensorflow
pip install -r requirements.txt
```

## Gerçek Zamanlı Kamera ile Kullanım

### Adım 1 — Kamera iznini ver (macOS)

macOS, Terminal uygulamasının kameraya erişmesine varsayılan olarak izin vermez.

1. **Sistem Ayarları** → **Gizlilik ve Güvenlik** → **Kamera** bölümüne git
2. Listede **Terminal**'i bul ve yanındaki kutucuğu aç
3. İzin verdikten sonra Terminal'i yeniden başlat

### Adım 2 — Kamera scriptini çalıştır

```bash
python src/live_demo.py
```

### Adım 3 — Kullanım

- Ekranın ortasında beliren **kutuya kumaşı tut**
- Model her 5 karede bir analiz yapar, sonuç sol üstte görünür:
  - Yeşil → **KUMAŞ SAĞLAM**
  - Kırmızı → **HATA tespit edildi**
  - Sarı → **Emin değil**
- Çıkmak için `q` tuşuna bas

## Tek Fotoğraf ile Test

`src/predict.py` dosyasını aç, `FOTOGRAF_YOLU` değişkenine test etmek istediğin fotoğrafın yolunu yaz:

```python
FOTOGRAF_YOLU = r"/tam/yol/kumas.jpg"
```

Sonra çalıştır:

```bash
python src/predict.py
```

## Model Eğitimi

Veri setini `data/MultiLabel_Dataset/` klasörüne koyduktan sonra:

```bash
python src/prepare_dataset.py   # Veri setini hazırla
python src/train.py              # Modeli eğit
```
