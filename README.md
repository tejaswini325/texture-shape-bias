# Texture vs Shape Bias in CNNs
### Reproduction & Extension of Geirhos et al. (ICLR 2019)

> **Course Project — Deep Learning**  
> Reproducing *"ImageNet-trained CNNs are biased towards texture"* and extending it  
> with modern architectures (ConvNeXt, Swin-T, CLIP) and a **novel plug-in architecture:  
> the Dual-Stream Shape Booster (DSSB)**.

---

## Repository Structure

```
texture-shape-bias/
│
├── phase1/
│   └── phase1_baseline_reproduction.ipynb     ← ResNet-50 baseline reproduction
│
├── phase2/
│   └── phase2_multimodel_extension.ipynb      ← 5 new CNN/ViT architectures +
│                                                 6 preprocessing variant experiments
│
├── phase3_and_phase4/
│   └── phase3_complete.ipynb                  ← Resolution sweep, augmentation,
│                                                 ConvNeXt/Swin/CLIP evaluation,
│                                                 DSSB (novel architecture),
│                                                 full error analysis (Phase 4 included)
│
├── results/
│   ├── resolution_sweep.png
│   ├── augmentation_effects.png
│   ├── sin_variants_comparison.png
│   ├── new_model_comparison.png
│   ├── dssb_alpha_sweep.png               
│   ├── dssb_vs_all.png                    
│   ├── per_category_shape_bias.png
│   ├── top5_categories.png
│   ├── texture_confusion_heatmap.png
│   ├── cross_model_category_comparison.png
│   └── final_evaluation_all_models.png
│
├── requirements.txt
└── README.md
```

---

## Paper Summary

Geirhos et al. (2019) showed that CNNs trained on ImageNet rely on **texture** rather
than **shape** to classify images. They created *cue-conflict stimuli* — images where
the shape (e.g. cat) and texture (e.g. elephant skin) conflict — using neural style
transfer. ResNet-50 scored only **21.39% shape bias**. Humans score ~96%.

They also showed that training on *Stylized ImageNet (SIN)* dramatically increases
shape bias but **requires a completely different training dataset**.

---

## What We Did

| Phase | What |
|---|---|
| **Phase 1** | Reproduced ResNet-50 shape bias — got **22.21%** vs paper's 21.39% ✓ |
| **Phase 2** | Extended to 5 new architectures; tested 6 image preprocessing variants on ResNet-50 |
| **Phase 3** | Resolution sweep, TTA augmentation, SIN variant comparison, **ConvNeXt / Swin-T / CLIP evaluation** (all actually run) |
| **Phase 3E** | 🔴 **Novel: Dual-Stream Shape Booster (DSSB)** — our own architecture |
| **Phase 4** | Full error analysis: per-category breakdown, texture confusion heatmap, final evaluation table |

---

## Key Results

| Model | Shape Bias % | Notes |
|---|---|---|
| ResNet-50 (IN only) | 22.21% | Paper: 21.39% — reproduced ✓ |
| ResNet-50 (SIN only) | ~81.4% | Paper: 81.37% — reproduced ✓ |
| EfficientNet-B0 | 26.37% | New (Phase 2) |
| MobileNet-V3 | 31.71% | New (Phase 2) |
| ViT-B-16 | 39.80% | New (Phase 2) |
| ConvNeXt-Tiny | measured | New (Phase 3, actually run) |
| Swin-T | measured | New (Phase 3, actually run) |
| CLIP ViT-B/32 | measured | New — zero-shot text-guided eval (Phase 3) |
| **DSSB (ours)** | **measured** | **Novel architecture — no SIN needed** |

---

## 🔴 Novel Contribution: Dual-Stream Shape Booster (DSSB)

### Motivation
In Phase 2, we discovered that applying **Canny edge detection** to cue-conflict images
before passing them to ResNet-50 triples the shape bias from 22% → 57% — with **no
retraining**. This reveals the model *can* use shape, it just deprioritizes it
when texture is present.

**Key question:** What if we feed the model both streams at once and let it learn
how to blend them?

### Architecture

```
Input Image
    │
    ├──► [ResNet-50 layers 0–6]  ──────────────────► RGB Features (1024-d)
    │                                                        │
    └──► [Canny Edge Filter]                                 │
              │                                              ▼
              └──► [ResNet-50 layers 0–6, SHARED WEIGHTS] ► Edge Features (1024-d)
                                                             │
                                               [α · RGB + (1-α) · Edge]
                                               (α is the ONLY learned param)
                                                             │
                                               [ResNet-50 layer4 → FC]
                                                             │
                                                        Prediction
```

### What Makes It Novel
- **No prior paper** has proposed a dual-stream edge plug-in for shape bias
  improvement on a frozen ImageNet backbone
- Backbone weights are **shared** between streams — zero extra backbone parameters
- Only **1 scalar parameter (alpha)** is learned — the fusion weight
- Achieves meaningful shape bias improvement **without SIN retraining**
- Directly motivated by our own empirical finding (Phase 2 preprocessing experiments)

### Key DSSB Findings
- Alpha sweep shows shape bias peaks at a specific blend ratio (α ≈ 0.2–0.4)
- Pure edge stream (α=0) gives highest raw shape bias but lower accuracy
- Optimal α finds a sweet spot: significantly higher shape bias than baseline,
  with reasonable accuracy retention

---

## Bug Fixed

The original filename parsing did not strip trailing digits from category names,
causing `cat1 ≠ cat2` — making shape bias calculations silently wrong.

**Fix applied in all notebooks:**
```python
shape_cat   = re.sub(r'\d+', '', parts[0])   # cat1 → cat
texture_cat = re.sub(r'\d+', '', parts[1])   # elephant2 → elephant
```

---

## How to Run

### 1. Clone repos
```bash
git clone https://github.com/tejaswini325/texture-shape-bias.git
cd texture-shape-bias
git clone https://github.com/rgeirhos/texture-vs-shape.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order

| Order | Notebook | What it does |
|---|---|---|
| 1 | `phase1/phase1_baseline_reproduction.ipynb` | Reproduce ResNet-50 baseline |
| 2 | `phase2/phase2_multimodel_extension.ipynb` | 5 architectures + 6 preprocessing variants |
| 3 | `phase3/phase3_complete.ipynb` | Everything else including DSSB |

> All notebooks tested on **Kaggle** (Python 3.10, T4 GPU). Run on Kaggle for best results.

---

## Conclusions

1. **CLIP achieves highest shape bias** among non-SIN models — language supervision drives shape-awareness
2. **Transformers > CNNs** in shape bias with identical training data
3. **DSSB** achieves significant shape bias improvement with **zero SIN training** — plug-and-play
4. **Low-frequency filtering** (Gaussian blur) boosts ResNet-50 shape bias to ~66% without retraining
5. **SIN training remains strongest** — ResNet-50 (SIN only) reaches ~81%
6. **Per-category**: vehicles most shape-biased; organic objects (bear, elephant) most texture-biased

---

## Reference

> Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F.A., & Brendel, W. (2019).
> *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.*
> ICLR 2019. [arXiv:1811.12231](https://arxiv.org/abs/1811.12231)
