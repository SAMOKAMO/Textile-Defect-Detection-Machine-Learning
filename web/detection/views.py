import time
import numpy as np
from datetime import timedelta

from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from . import model_loader

ESIK = 0.6


def index(request):
    return render(request, 'detection/index.html')


@csrf_exempt
def predict_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Sadece POST istekleri kabul edilir.'}, status=405)

    if 'image' not in request.FILES:
        return JsonResponse({'error': "İstekte 'image' dosyası bulunamadı."}, status=400)

    if model_loader.model is None:
        return JsonResponse({'error': 'Model henüz yüklenmedi.'}, status=503)

    try:
        import cv2
        from tensorflow.keras.applications.resnet50 import preprocess_input

        image_file = request.FILES['image']
        img = Image.open(image_file).convert('RGB')
        img = img.resize((512, 512))
        img_array = np.array(img).astype(np.float32)

        # Texture gate: reject uniform/out-of-distribution inputs
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        texture_std = float(gray.std())
        if texture_std < 10:
            return JsonResponse({
                'error': 'low_texture',
                'message': 'Görüntü çok düzgün — kumaşı daha yakına tutun.',
                'texture_std': round(texture_std, 1),
            }, status=422)

        img_array = preprocess_input(img_array)

        t0 = time.perf_counter()
        tahminler = model_loader.model.predict(
            np.expand_dims(img_array, 0), verbose=0
        )[0]
        inference_ms = round((time.perf_counter() - t0) * 1000, 1)

        sonuclar = {
            sinif: float(olasilik)
            for sinif, olasilik in zip(model_loader.siniflar, tahminler)
        }

        aktif = [s for s, p in sonuclar.items() if p >= ESIK]

        # Log every prediction that has at least one class above threshold
        if aktif:
            from .models import DefectLog
            DefectLog.objects.create(
                siniflar=sonuclar,
                aktif_hatalar=aktif,
                esik=ESIK,
            )

        return JsonResponse({'tahminler': sonuclar, 'esik': ESIK, 'inference_ms': inference_ms})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def log_api(request):
    """Return the last N defect log entries as JSON."""
    from .models import DefectLog

    try:
        limit = min(int(request.GET.get('limit', 20)), 200)
    except ValueError:
        limit = 20

    entries = DefectLog.objects.all()[:limit]
    data = [
        {
            'id':            e.id,
            'timestamp':     e.timestamp.isoformat(),
            'siniflar':      e.siniflar,
            'aktif_hatalar': e.aktif_hatalar,
            'esik':          e.esik,
            'kaynak':        e.kaynak,
        }
        for e in entries
    ]
    return JsonResponse({'log': data, 'count': len(data)})


def summary_api(request):
    """Aggregate defect counts for a given time window.

    ?window=15m   last 15 minutes
    ?window=1h    last 1 hour  (default)
    ?window=shift last 8 hours
    """
    from .models import DefectLog

    WINDOWS = {'15m': 15, '1h': 60, 'shift': 480}
    window = request.GET.get('window', '1h')
    minutes = WINDOWS.get(window, 60)

    since = timezone.now() - timedelta(minutes=minutes)
    logs = DefectLog.objects.filter(timestamp__gte=since)

    total = logs.count()

    # Per-class active detection counts
    class_counts = {}
    for entry in logs:
        for sinif in entry.aktif_hatalar:
            class_counts[sinif] = class_counts.get(sinif, 0) + 1

    # Detections per minute (for rolling chart)
    per_minute = {}
    for entry in logs:
        key = entry.timestamp.strftime('%H:%M')
        per_minute[key] = per_minute.get(key, 0) + 1

    return JsonResponse({
        'window':              window,
        'since':               since.isoformat(),
        'total_predictions':   total,
        'class_counts':        class_counts,
        'detections_per_minute': per_minute,
    })
