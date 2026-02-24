# Tasks

---

## Task 1: Fix FileNotFoundError — veri_etiketleri.csv Missing
**Status: DONE ✅** (2026-02-24)

1. `src/live_demo.py` — hard `raise` → `try/except` + fallback class list.
2. `web/detection/model_loader.py` — `'Vertical'` → `'vertical'` capitalisation bug.

---

## Task 2: Django Web App — Base Layer
**Status: DONE ✅** (2026-02-24)

| File | Status |
|---|---|
| All Python backend files | ✅ |
| `camera.js` | ✅ Written |
| Django 6.0.2 + Pillow 12.1.1 installed | ✅ |
| `manage.py check` → 0 issues | ✅ |
| Fix 1: apps.py graceful degradation | ✅ |
| Fix 2: model_loader assertion | ✅ |
| Fix 3: camera.js 4s fetch timeout | ✅ |

**Remaining:**
- [ ] Step D — `python web/manage.py runserver` live test
- [ ] Step E — Browser test at http://localhost:8000

---

## Task 3: Defect Event Log
**Status: DONE ✅** (2026-02-24)
**Academic value:** Audit trail; turns "my model works" into provable claim.
**Effort:** ~1 day

### What
Every prediction that crosses the alert threshold is written to a SQLite
database (no extra infra, ships with Python/Django). Each record stores:
- timestamp, defect classes detected, per-class confidence scores
- frame snapshot path (optional JPEG thumbnail)
- source: 'web' or 'live_demo'

### Why
- Enables shift summary report (defect count/hour, false positive rate)
- Creates ground-truth dataset for threshold calibration section of thesis
- Answers committee question: "How do you know it works in practice?"

### Steps
- [x] `detection/models.py` — DefectLog model created
- [x] `settings.py` — DATABASES (SQLite) + contenttypes app added
- [x] `python web/manage.py migrate` — table created (detection.0001_initial)
- [x] `views.py` — predict_api writes log when aktif_hatalar non-empty
- [x] `/api/log/?limit=N` endpoint — returns last N records as JSON
- [x] `/api/summary/?window=15m|1h|shift` — counts per class + per-minute buckets
- [x] DB write/read verified with 3 test records → cleaned

---

## Task 4: Temporal Voting (False-Positive Suppression)
**Status: PENDING**
**Academic value:** Core thesis contribution — bridges raw ML output to
industrial action. Addresses false-positive problem systematically.
**Effort:** ~half day (pure JS, no backend changes)

### What
A sliding window of the last 5 predictions per class.
A class is only "confirmed" when it wins a majority vote (≥ 3/5).

```
Raw model output  → [1, 0, 1, 1, 0]  →  vote = 3/5  →  ALERT
                  → [1, 1, 1, 1, 1]  →  vote = 5/5  →  SUGGEST HALT
```

### Why
Eliminates single-frame noise. Dramatically reduces false alert rate.
Provides two distinct action levels (alert vs. suggest halt) tied to
confidence — defensible design in thesis.

### Steps
- [ ] Add `PredictionHistory` class in `camera.js` (ring buffer, size=5)
- [ ] Compute vote score per class before calling `updateUI()`
- [ ] Map vote score to three response levels:
  - `vote < 3` → LOG only (no UI change, only backend write)
  - `vote >= 3` → ALERT (yellow/red banner, existing behavior)
  - `vote == 5 AND max_confidence > 0.85` → SUGGEST HALT (new banner state)
- [ ] Add 'suggest_halt' CSS state to banner + hedef-cerceve
- [ ] Pass `vote_scores` alongside raw predictions to log endpoint

---

## Task 5: Supervisor Dashboard
**Status: PENDING**
**Academic value:** Human-AI collaboration — shows system is designed for
real users, not just benchmark metrics. Strongest visual impact for committee.
**Effort:** ~2 days

### What
A second page (`/dashboard/`) with three panels:

**Panel 1 — Rolling Defect Rate (last 15 min)**
- Line chart (Chart.js, CDN, no install needed)
- X-axis: time (1-min buckets), Y-axis: detections/min
- One line per defect class + one for 'all defects'
- Data source: `/api/summary/?window=15m`

**Panel 2 — Defect Type Distribution (current shift)**
- Doughnut chart
- Data source: `/api/summary/?window=shift`

**Panel 3 — Event Table**
- Last 20 logged events: time, defect type, confidence, vote score
- Color-coded rows: green (defect_free), red (defect), yellow (borderline)
- Data source: `/api/log/?limit=20`
- Auto-refreshes every 10 seconds

### Steps
- [ ] Add `dashboard` view + template
- [ ] Wire `/dashboard/` URL
- [ ] Implement `/api/summary/` with `window` parameter
- [ ] Implement `/api/log/` with `limit` parameter
- [ ] Build `dashboard.html` with Chart.js panels (CDN link, no install)
- [ ] Build `dashboard.js` with 10s polling loop
- [ ] Add nav link from main camera page to dashboard

---

## Task 6: Per-Class Threshold Calibration Table
**Status: PENDING**
**Academic value:** Novel analytical contribution. Shows mastery of ML
deployment beyond accuracy metrics.
**Effort:** ~half day (analysis + one config change)

### What
Different defect classes have different consequences and different
model confidence distributions. A single threshold (0.6) is not optimal.

| Class | Severity | Suggested Threshold | Rationale |
|---|---|---|---|
| hole | Critical (structural) | 0.75 | False negatives very costly |
| horizontal | High | 0.65 | Directional defect, clear pattern |
| vertical | High | 0.65 | Same as horizontal |
| stain | Medium | 0.60 | Cosmetic, lower risk |
| lines | Medium | 0.60 | Cosmetic, lower risk |
| defect_free | — | 0.55 | Lower bar for "clear" confirmation |

### Steps
- [ ] Add `SINIF_ESIKLERI` dict to `src/config.py`
- [ ] Pass per-class thresholds in `/api/predict/` response
- [ ] Update `camera.js` `updateUI()` to use per-class threshold
- [ ] Update `model_loader.py` to expose threshold config
- [ ] Document calibration rationale in thesis (Section: Deployment Design)

---

## Task 7: Supervisor Override + Feedback Loop
**Status: PENDING — Lower priority, implement if time allows**
**Academic value:** Human-in-the-loop; closes the feedback cycle.

### What
When the system suggests a halt, supervisor sees two buttons:
- "Confirm — Stop Machine" → logs override_type='confirm'
- "Dismiss — False Alarm"  → logs override_type='dismiss'

Override log feeds into threshold calibration analysis for thesis.

### Steps
- [ ] Add `override` field to `DefectLog` model
- [ ] Add `/api/override/<log_id>/` POST endpoint
- [ ] Render confirm/dismiss buttons in camera.js when vote==5
- [ ] Aggregate dismiss rate per class in `/api/summary/` response

---

## Implementation Order

```
Task 3 (Log)  →  Task 4 (Voting)  →  Task 5 (Dashboard)  →  Task 6 (Thresholds)
   1 day            0.5 day              2 days                  0.5 day
```

Task 3 is the foundation — Tasks 5 and 7 depend on it.
Task 4 is independent — can be done in parallel with Task 3.

---

## Optional (future)
- [ ] Task 7: Supervisor Override (if time allows)
- [ ] Add raw images to `data/Temiz_Veri_Seti/<ClassName>/` and run `prepare_dataset.py`
- [ ] Retrain with more data if available
