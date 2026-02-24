# Lessons Learned

## L-001: Defend Against Missing Data Files at Script Startup

**Date:** 2026-02-24
**Trigger:** `live_demo.py` raised `FileNotFoundError` for `veri_etiketleri.csv`
on a machine where the raw dataset was never placed (only the trained model was
present).

**Pattern:**
> Scripts that `raise` at the top level for missing *optional* files (files
> whose only purpose is to derive class names that could be hardcoded as a
> fallback) will always break on a clean checkout where only the model is
> present.

**Rule:**
- Any file that is NOT the model itself (weights) should be wrapped in
  `try/except FileNotFoundError` with a documented fallback.
- The model file *should* remain a hard guard — if the model is missing,
  inference is impossible.
- The class list is derivable from the model's final layer output shape and can
  be hardcoded as a fallback when the CSV is unavailable.

**Fix pattern:**
```python
# BEFORE (brittle)
if not Path(CSV_YOLU).exists():
    raise FileNotFoundError(f"CSV bulunamadı: {CSV_YOLU}")

# AFTER (resilient)
try:
    df = pd.read_csv(CSV_YOLU)
    SINIFLAR = sorted({e for row in df['etiketler'].dropna()
                       for e in str(row).split()})
except FileNotFoundError:
    SINIFLAR = ['defect_free', 'hole', 'horizontal', 'lines', 'stain', 'vertical']
    print(f"WARNING: CSV not found, using fallback classes: {SINIFLAR}")
```

---

## L-002: Always Verify Class Count Against Model Output Shape — Now Enforced in Code

**Date:** 2026-02-24
**Trigger:** Fallback class list in `model_loader.py` had 'Vertical' (capital V)
while `prepare_dataset.py` normalises all class names to lowercase.

**Rule:**
- After defining a class list (from CSV or fallback), assert
  `len(classes) == model.output_shape[-1]` before proceeding.
- Class names in fallback lists must follow the same normalisation as
  `prepare_dataset.py`: `folder_name.lower().replace(' ', '_')`.

---

## L-004: OOD Behavior — Sigmoid Classifiers Have No "I Don't Know"

**Date:** 2026-02-25
**Trigger:** Model predicted `hole: 99.1%` on a plain gray sweatshirt.

**Root Cause:**
ResNet50 + sigmoid + binary cross-entropy has no out-of-distribution (OOD)
rejection mechanism. On any input — including solid colors, arbitrary fabric,
or non-textile objects — the model always outputs values in [0,1]. These
look like probabilities but are not calibrated for OOD inputs.

**Verified experimentally:**
| Input          | Top prediction       |
|----------------|----------------------|
| Solid gray     | hole: 99.1%          |
| Solid white    | hole: 100%           |
| Solid black    | lines: 99.7%         |
| Random noise   | vertical+hole: 100%  |

**Fix Applied:**
Texture gate in `views.py` — compute grayscale std of the input image.
If `std < 10`, return `{"error": "low_texture"}` (HTTP 422) without
running the model. Camera.js shows "Kumaşı daha yakın tutun".

**Thesis implication:**
This is a known limitation of discriminative classifiers and worth
documenting explicitly. The model assumes the input is from the training
distribution (close-up textile fabric). A proper fix would require
training with negative examples or adding an explicit OOD detection head.

**Rule:**
Always add a pre-inference input validity check. Never trust sigmoid
output on arbitrary inputs.

---

## L-003: Plan Before Code — CLAUDE.md Workflow

**Date:** 2026-02-24

**Rule:**
- Always write the plan to `tasks/todo.md` before touching any source file.
- Never start implementation without verifying the root cause first.
- Record every correction or surprise in `tasks/lessons.md` immediately.
