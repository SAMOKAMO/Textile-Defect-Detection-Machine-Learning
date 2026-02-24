'use strict';

// ─── DOM refs ────────────────────────────────────────────────────────────────
const video          = document.getElementById('video');
const canvas         = document.getElementById('canvas');
const ctx            = canvas.getContext('2d');
const sinifListesi   = document.getElementById('sinif-listesi');
const durumYazi      = document.getElementById('durum-yazi');
const durumBanner    = document.getElementById('durum-banner');
const hedefCerceve   = document.getElementById('hedef-cerceve');
const fpsGoster      = document.getElementById('fps-goster');
const tahminSayac    = document.getElementById('tahmin-sayac');
const baglantiGoster = document.getElementById('baglanti-goster');
const tabKamera      = document.getElementById('tab-kamera');
const tabYukle       = document.getElementById('tab-yukle');
const yukleAlan      = document.getElementById('yukle-alan');
const dosyaInput     = document.getElementById('dosya-input');
const onizleme       = document.getElementById('onizleme');
const baskaGorsel    = document.getElementById('baska-gorsel');
const kameraBilgi    = document.getElementById('kamera-bilgi');

// ─── Config ──────────────────────────────────────────────────────────────────
const INTERVAL_MS  = 500;
const JPEG_QUALITY = 0.82;
const API_URL      = '/api/predict/';
const WINDOW_SIZE  = 5;    // frames in rolling window
const VOTE_MIN     = 3;    // votes needed to confirm a class

// ─── State ───────────────────────────────────────────────────────────────────
let currentMode      = 'kamera';
let cameraStream     = null;
let cameraIntervalId = null;
let isRequesting     = false;
let requestCount     = 0;
let lastSuccessTime  = null;
const history        = {};   // { className: [prob, ...] }  max WINDOW_SIZE

// ─── Helpers ─────────────────────────────────────────────────────────────────
function getCookie(name) {
  const match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
  return match ? decodeURIComponent(match[1]) : null;
}

function setDurum(text, cssClass) {
  durumYazi.textContent  = text;
  durumBanner.className  = 'durum ' + cssClass;
  hedefCerceve.className = 'hedef-cerceve ' + cssClass;
}

// ─── Ring buffer helpers ──────────────────────────────────────────────────────
function pushHistory(sinif, prob) {
  if (!history[sinif]) history[sinif] = [];
  history[sinif].push(prob);
  if (history[sinif].length > WINDOW_SIZE) history[sinif].shift();
}

function votes(sinif, esik) {
  return (history[sinif] || []).filter(p => p >= esik).length;
}

function smoothed(sinif) {
  const h = history[sinif] || [];
  if (!h.length) return 0;
  return h.reduce((a, b) => a + b, 0) / h.length;
}

function framesCollected() {
  const vals = Object.values(history);
  return vals.length ? vals[0].length : 0;
}

function clearHistory() {
  Object.keys(history).forEach(k => delete history[k]);
}

// ─── Shared result renderer ───────────────────────────────────────────────────
// useVoting=true  → camera mode (ring buffer + majority vote, warmup phase)
// useVoting=false → upload mode (single frame, raw threshold)
function renderResults(tahminler, esik, inference_ms, useVoting) {
  lastSuccessTime = Date.now();
  requestCount   += 1;
  if (inference_ms !== undefined) fpsGoster.textContent = inference_ms;
  tahminSayac.textContent = requestCount;

  if (useVoting) {
    Object.entries(tahminler).forEach(([s, p]) => pushHistory(s, p));
  }

  const collected = useVoting ? framesCollected() : WINDOW_SIZE;
  const warming   = useVoting && collected < WINDOW_SIZE;

  // Sort descending by display value
  const entries = Object.keys(tahminler)
    .map(sinif => [sinif, useVoting ? smoothed(sinif) : tahminler[sinif]])
    .sort((a, b) => b[1] - a[1]);

  sinifListesi.innerHTML = entries.map(([sinif, avg]) => {
    // In upload mode, treat "above esik" as a confirmed vote
    const v      = useVoting ? votes(sinif, esik) : (avg >= esik ? VOTE_MIN : 0);
    const onaylı = v >= VOTE_MIN;
    const pct    = (avg * 100).toFixed(1);
    const bw     = (avg * 100).toFixed(2);

    const cCls = onaylı ? (sinif === 'defect_free' ? 'aktif-saglam' : 'aktif-hata') : '';
    let bCls = '';
    if (onaylı)                bCls = sinif === 'defect_free' ? 'yuksek-saglam' : 'yuksek-hata';
    else if (avg >= esik * 0.6) bCls = 'orta';

    // Camera mode: vote dots   Upload mode: plain percentage
    let badge;
    if (useVoting) {
      const dc = sinif === 'defect_free' ? 'var(--yesil)' : 'var(--kirmizi)';
      badge = Array.from({ length: WINDOW_SIZE }, (_, i) =>
        `<span style="color:${i < v ? dc : '#3a3a3a'};font-size:0.7em">●</span>`
      ).join('') + `&nbsp;${pct}%`;
    } else {
      badge = `${pct}%`;
    }

    return `
      <div class="sinif-kart ${cCls}">
        <div class="sinif-bilgi">
          <span class="sinif-adi">${sinif.replace(/_/g, ' ')}</span>
          <span class="sinif-yuzde">${badge}</span>
        </div>
        <div class="progress-track">
          <div class="progress-bar ${bCls}" style="width:${bw}%"></div>
        </div>
      </div>`;
  }).join('');

  // Banner — defer decision until window is full in camera mode
  if (warming) {
    setDurum(`Analiz ediliyor... (${collected}/${WINDOW_SIZE} kare)`, 'belirsiz');
    return;
  }

  const hatalar = entries
    .filter(([s]) => s !== 'defect_free')
    .filter(([s, avg]) => useVoting ? votes(s, esik) >= VOTE_MIN : avg >= esik)
    .map(([s]) => s);
  const saglam = useVoting
    ? votes('defect_free', esik) >= VOTE_MIN
    : (tahminler['defect_free'] || 0) >= esik;

  if (hatalar.length > 0) {
    const isimler = hatalar.map(s => s.replace(/_/g, ' ')).join(' + ').toUpperCase();
    setDurum('HATA: ' + isimler, 'hatali');
  } else if (saglam) {
    setDurum('KUMAŞ SAĞLAM ✓', 'saglam');
  } else {
    setDurum('Belirsiz', 'belirsiz');
  }
}

// ─── Camera: capture & predict loop ──────────────────────────────────────────
function captureAndPredict() {
  if (isRequesting || video.readyState < 2) return;
  isRequesting = true;

  // Center-crop 16:9 → 1:1 square (avoids aspect-ratio distortion)
  const vw   = video.videoWidth  || canvas.width;
  const vh   = video.videoHeight || canvas.height;
  const side = Math.min(vw, vh);
  ctx.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    if (!blob) { isRequesting = false; return; }

    const fd = new FormData();
    fd.append('image', blob, 'frame.jpg');
    const controller = new AbortController();
    const tid = setTimeout(() => controller.abort(), 4000);

    fetch(API_URL, {
      method:  'POST',
      body:    fd,
      headers: { 'X-CSRFToken': getCookie('csrftoken') || '' },
      signal:  controller.signal,
    })
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(data => {
        if (data.error === 'low_texture') { setDurum('Kumaşı daha yakın tutun', 'belirsiz'); return; }
        if (data.error) throw new Error(data.error);
        renderResults(data.tahminler, data.esik, data.inference_ms, true);
      })
      .catch(err => setDurum(
        err.name === 'AbortError' ? 'Sunucu yanıt vermiyor (4s timeout)' : 'Sunucu hatası: ' + err.message,
        'belirsiz'
      ))
      .finally(() => { clearTimeout(tid); isRequesting = false; });
  }, 'image/jpeg', JPEG_QUALITY);
}

// ─── Upload: single image predict ────────────────────────────────────────────
function predictImage(file) {
  setDurum('Analiz ediliyor...', 'belirsiz');
  sinifListesi.innerHTML = '<p class="bekleme-mesaji">Tahmin hesaplanıyor...</p>';

  const fd = new FormData();
  fd.append('image', file, file.name);
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 8000);

  fetch(API_URL, {
    method:  'POST',
    body:    fd,
    headers: { 'X-CSRFToken': getCookie('csrftoken') || '' },
    signal:  controller.signal,
  })
    .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(data => {
      if (data.error === 'low_texture') {
        setDurum('Görüntü çok düzgün — kumaş görseli seçin', 'belirsiz');
        return;
      }
      if (data.error) throw new Error(data.error);
      renderResults(data.tahminler, data.esik, data.inference_ms, false);
    })
    .catch(err => setDurum(
      err.name === 'AbortError' ? 'Sunucu yanıt vermiyor (8s timeout)' : 'Sunucu hatası: ' + err.message,
      'belirsiz'
    ))
    .finally(() => clearTimeout(tid));
}

// ─── Mode switching ───────────────────────────────────────────────────────────
function switchMode(mode) {
  if (mode === currentMode) return;
  currentMode = mode;

  tabKamera.classList.toggle('aktif', mode === 'kamera');
  tabYukle.classList.toggle('aktif',  mode === 'yukle');

  if (mode === 'kamera') {
    // Hide upload UI
    yukleAlan.style.display   = 'none';
    onizleme.style.display    = 'none';
    baskaGorsel.style.display = 'none';
    onizleme.src = '';
    dosyaInput.value = '';

    // Show camera UI
    video.style.display        = '';
    hedefCerceve.style.display = '';
    kameraBilgi.style.display  = '';

    // Reset voting state
    clearHistory();
    setDurum(`Analiz ediliyor... (0/${WINDOW_SIZE} kare)`, 'belirsiz');
    sinifListesi.innerHTML = '<p class="bekleme-mesaji">Tahmin bekleniyor...</p>';

    // Restart capture loop
    if (!cameraStream) {
      startCamera();
    } else if (cameraIntervalId === null) {
      cameraIntervalId = setInterval(captureAndPredict, INTERVAL_MS);
    }

  } else {
    // mode === 'yukle' — pause capture loop, keep stream alive
    if (cameraIntervalId !== null) {
      clearInterval(cameraIntervalId);
      cameraIntervalId = null;
    }

    // Hide camera UI, show upload UI
    video.style.display        = 'none';
    hedefCerceve.style.display = 'none';
    kameraBilgi.style.display  = 'none';
    yukleAlan.style.display    = '';

    setDurum('Görsel yükleyin', 'bekliyor');
    sinifListesi.innerHTML = '<p class="bekleme-mesaji">Tahmin bekleniyor...</p>';
  }
}

// ─── File input handler ───────────────────────────────────────────────────────
dosyaInput.addEventListener('change', () => {
  const file = dosyaInput.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  onizleme.src = url;
  onizleme.onload = () => URL.revokeObjectURL(url);

  yukleAlan.style.display   = 'none';
  onizleme.style.display    = '';
  baskaGorsel.style.display = '';

  predictImage(file);
});

baskaGorsel.addEventListener('click', () => {
  onizleme.style.display    = 'none';
  baskaGorsel.style.display = 'none';
  yukleAlan.style.display   = '';
  onizleme.src  = '';
  dosyaInput.value = '';
  setDurum('Görsel yükleyin', 'bekliyor');
  sinifListesi.innerHTML = '<p class="bekleme-mesaji">Tahmin bekleniyor...</p>';
  hedefCerceve.className = 'hedef-cerceve';
});

// ─── Tab listeners ────────────────────────────────────────────────────────────
tabKamera.addEventListener('click', () => switchMode('kamera'));
tabYukle.addEventListener( 'click', () => switchMode('yukle'));

// ─── Connection staleness ticker ──────────────────────────────────────────────
setInterval(() => {
  if (lastSuccessTime === null) {
    baglantiGoster.textContent = '—';
    baglantiGoster.style.color = '';
    return;
  }
  const secs = Math.round((Date.now() - lastSuccessTime) / 1000);
  baglantiGoster.textContent = secs + 's';
  baglantiGoster.style.color =
    secs <= 2 ? 'var(--yesil)' : secs <= 5 ? 'var(--sari)' : 'var(--kirmizi)';
}, 1000);

// ─── Camera init ──────────────────────────────────────────────────────────────
function startCamera() {
  navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
    .then(stream => {
      cameraStream = stream;
      video.srcObject = stream;
      setDurum(`Analiz ediliyor... (0/${WINDOW_SIZE} kare)`, 'belirsiz');
      cameraIntervalId = setInterval(captureAndPredict, INTERVAL_MS);
    })
    .catch(err => {
      setDurum('Kamera erişim hatası: ' + err.message, 'hatali');
      sinifListesi.innerHTML = `<div class="hata-mesaji">
        Kamera izni verilmedi veya cihaz bulunamadı.<br>
        Tarayıcı adres çubuğundan kamera iznini kontrol edin.
      </div>`;
    });
}

startCamera();
