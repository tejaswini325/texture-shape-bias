# Texture vs Shape Bias in CNNs
### Reproduction & Extension of Geirhos et al. (ICLR 2019)

> **Course Project — Deep Learning**
> Reproducing the paper *"ImageNet-trained CNNs are biased towards texture"* and extending it with modern architectures including ConvNeXt, Swin Transformer, and CLIP.

---

## Repository Structure

```
texture-shape-bias/
│
├── phase1/
│   └── phase1_baseline_reproduction.ipynb     ← ResNet-50 reproduction
│
├── phase2/
│   ├── phase2_multimodel_extension.ipynb      ← 5 new architectures tested
│   └── phase2_sin_visualization.ipynb         ← SIN model decision visualization
│
├── phase3/
│   └── phase3_hyperparameter_tuning.ipynb     ← Resolution sweep, augmentation,
│                                                 ConvNeXt / Swin / CLIP comparison
│
├── phase4/
│   └── phase4_error_analysis_evaluation.ipynb ← Per-category analysis, heatmap,
│                                                 final evaluation table
│
├── results/
│   ├── resolution_sweep.png
│   ├── augmentation_effects.png
│   ├── sin_variants_comparison.png
│   ├── new_model_comparison.png
│   ├── per_category_shape_bias.png
│   ├── per_category_new_models.png
│   ├── top5_categories.png
│   ├── texture_confusion_heatmap.png
│   ├── cross_model_category_comparison.png
│   └── final_evaluation_all_models.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Paper Summary

Geirhos et al. (2019) showed that CNNs trained on ImageNet rely on **texture** rather than **shape** to classify images. They created *cue-conflict stimuli* — images where the shape (e.g. cat) and texture (e.g. elephant skin) belong to different classes using neural style transfer. ResNet-50 classified these based on texture 78% of the time, scoring only **21.39% shape bias**. Humans score ~96%.

They also showed that training on *Stylized ImageNet (SIN)* — where all textures are randomized — dramatically increases shape bias and improves robustness.

---

## What We Did

| Phase | What |
|---|---|
| **Phase 1** | Reproduced ResNet-50 shape bias — got **22.21%** vs paper's 21.39% |
| **Phase 2** | Extended to 5 new models: EfficientNet-B0, MobileNet-V3, DenseNet-121, RegNet-Y-400MF, ViT-B-16 |
| **Phase 3** | Hyperparameter tuning (resolution sweep, test-time augmentation) + **3 recent models**: ConvNeXt-Tiny, Swin-T, CLIP ViT-B/32 |
| **Phase 4** | Full error analysis: per-category breakdown, texture confusion heatmap, cross-model comparison, final evaluation table |

---

## Key Results

| Model | Shape Bias % | Notes |
|---|---|---|
| ResNet-50 (IN only) | 22.21% | Paper: 21.39% — reproduced ✓ |
| ResNet-50 (SIN only) | ~81.4% | Paper: 81.37% — reproduced ✓ |
| EfficientNet-B0 | 26.37% | New |
| MobileNet-V3 | 31.71% | New |
| ViT-B-16 | 39.80% | New — nearly 2× ResNet-50 |
| ConvNeXt-Tiny | ~27% | New (2022 model) |
| Swin-T | ~33% | New (2021 model) |
| **CLIP ViT-B/32** | **~46%** | **Best among non-SIN models** |

**Best model: CLIP ViT-B/32** — language supervision forces semantic, shape-aware representations, achieving the highest shape bias without any SIN training.

---

## How to Run

### 1. Clone this repo

```bash
git clone https://github.com/tejaswini325/texture-shape-bias.git
cd texture-shape-bias
```

### 2. Clone the texture-vs-shape repo (required for stimuli + SIN models)

```bash
git clone https://github.com/rgeirhos/texture-vs-shape.git
```

This provides the cue-conflict stimuli and `load_pretrained_models.py`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run notebooks in order

| Order | Notebook | What it does |
|---|---|---|
| 1 | `phase1/phase1_baseline_reproduction.ipynb` | Reproduce ResNet-50 result |
| 2 | `phase2/phase2_multimodel_extension.ipynb` | Run 5 new architectures |
| 3 | `phase2/phase2_sin_visualization.ipynb` | Visualize SIN model decisions |
| 4 | `phase3/phase3_hyperparameter_tuning.ipynb` | Resolution, augmentation, new models |
| 5 | `phase4/phase4_error_analysis_evaluation.ipynb` | Full error analysis |

> All notebooks were developed and tested on **Kaggle** (Python 3.10, GPU T4).
> Run on Kaggle for best results — select GPU accelerator.

### 5. Save result plots

After running Phase 3 and Phase 4, move all generated `.png` files into the `results/` folder before committing.

---

## Bug Fixed (Phase 3)

The original filename parsing code did not strip trailing digits from category names, causing `cat1` and `cat2` to be treated as different categories — making the shape bias calculation incorrect.

**Fix applied in all Phase 3+ code:**
```python
shape_cat   = re.sub(r'\d+', '', parts[0])   # cat1 → cat
texture_cat = re.sub(r'\d+', '', parts[1])   # elephant2 → elephant
```

---

## Conclusions

1. **CLIP achieves highest shape bias (~46%)** among all non-SIN models — language supervision is key.
2. **Transformers > CNNs** in shape bias even with identical training data.
3. **Grayscale preprocessing** boosts shape bias by ~4% — removing color forces shape reliance.
4. **SIN training remains the strongest intervention** — ResNet-50 (SIN only) reaches 81%.
5. **Per-category**: vehicles (bicycle, airplane) are most shape-biased; organic objects (bear, elephant) most texture-biased.

---

## Reference

> Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F.A., & Brendel, W. (2019).
> *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.*
> ICLR 2019. [arXiv:1811.12231](https://arxiv.org/abs/1811.12231)
