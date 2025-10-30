# Thermal Anomaly Detection System
**Euclidean Technologies - Advanced Man-Made Anomaly Detection**

A production-grade deep learning system for detecting man-made thermal anomalies in satellite imagery (400–2500 nm) using Swin Transformer U-Net architecture.

---

## 🎯 Overview

This system detects buildings, vehicles, industrial facilities, and other man-made structures in thermal and hyperspectral satellite imagery through a multi-stage pipeline combining statistical methods, deep learning, and material characterization.

**Key Capabilities:**
- Multi-sensor support (Landsat, Sentinel-2, ASTER, MODIS)
- Dual-mode processing: Fast statistical + Deep learning
- Multi-modal fusion (thermal + hyperspectral)
- Material characterization via emissivity and spectral analysis
- Scalable from edge devices to cloud clusters

---

## 🔄 Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE OVERVIEW                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────┐
│ 1. DATA INPUT  │
│   Multi-Sensor │
└────────┬───────┘
         │
         ├─► Landsat (TIF, HE5)
         ├─► Sentinel-2 (TIF)
         ├─► ASTER (HE5, MAT)
         └─► MODIS (HDF)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING & NORMALIZATION                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │Format Parser │───►│Band Extraction│───►│ Validation   │             │
│  │(he5/mat/tif) │    │ (thermal/HSI) │    │ (NaN, range) │             │
│  └──────────────┘    └──────────────┘    └──────┬───────┘             │
│                                                   │                      │
│  ┌────────────────────────────────────────────────▼─────────────────┐  │
│  │ ROBUST NORMALIZATION (per-band)                                  │  │
│  │   μ_b = median(X_b)                                              │  │
│  │   σ_b = median(|X_b − μ_b|) + ε                                 │  │
│  │   Z_b = (X_b − μ_b) / σ_b                                       │  │
│  └────────────────────────────────────────────────┬─────────────────┘  │
│                                                    │                     │
│  ┌────────────────────────────────────────────────▼─────────────────┐  │
│  │ LOCAL BACKGROUND MODELING (5×5 km windows)                       │  │
│  │   • Terrain-adjusted baseline                                    │  │
│  │   • Diurnal correction                                           │  │
│  │   • Spatial context preservation                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
         ┌────────────────────────────────┴────────────────────────────┐
         │                                                              │
         ▼                                                              ▼
┌─────────────────────┐                                    ┌─────────────────────┐
│ 3A. FAST PATH       │                                    │ 3B. DEEP LEARNING   │
│  (Statistical)      │                                    │  (Neural Network)   │
├─────────────────────┤                                    ├─────────────────────┤
│                     │                                    │                     │
│ THERMAL THRESHOLD:  │                                    │ SWIN TRANSFORMER    │
│  τ_build = μ+2σ    │                                    │      U-NET          │
│  τ_vehicle = μ+3σ  │                                    │                     │
│  τ_indust = μ+4σ   │                                    │ ┌─────────────────┐ │
│                     │                                    │ │  ENCODER        │ │
│ mask = (T > τ)     │                                    │ │ ┌─────────────┐ │ │
│                     │                                    │ │ │PatchEmbed   │ │ │
│ HSI RX DETECTOR:    │                                    │ │ │  (4×4)      │ │ │
│  z = (x−μ) ⊘ σ    │                                    │ │ └──────┬──────┘ │ │
│  s_RX = ||z||²    │                                    │ │        │        │ │
│                     │                                    │ │ ┌──────▼──────┐ │ │
│ PCA GENERATIVE:     │                                    │ │ │ Swin Stage1 │ │ │
│  x̂ = P_k·P_k^T·x   │                                    │ │ │ depth=2,h=3 │ │ │
│  s_RE = ||x−x̂||²  │                                    │ │ └──────┬──────┘ │ │
│                     │                                    │ │   PatchMerge   │ │
│ COMBINED:           │                                    │ │ ┌──────▼──────┐ │ │
│  s = 0.6·RX+0.4·RE │                                    │ │ │ Swin Stage2 │ │ │
│  mask = (s>μ+6σ)   │                                    │ │ │ depth=2,h=6 │ │ │
│                     │                                    │ │ └──────┬──────┘ │ │
│ Speed: ~100 FPS CPU │                                    │ │   PatchMerge   │ │
│                     │                                    │ │ ┌──────▼──────┐ │ │
└──────────┬──────────┘                                    │ │ │ Swin Stage3 │ │ │
           │                                               │ │ │ depth=6,h=12│ │ │
           │                                               │ │ └──────┬──────┘ │ │
           │                                               │ │   PatchMerge   │ │
           │                                               │ │ ┌──────▼──────┐ │ │
           │                                               │ │ │ Swin Stage4 │ │ │
           │                                               │ │ │ depth=2,h=24│ │ │
           │                                               │ │ └──────┬──────┘ │ │
           │                                               │ └────────┼────────┘ │
           │                                               │          │          │
           │                                               │ ┌────────▼────────┐ │
           │                                               │ │  DECODER        │ │
           │                                               │ │ ┌─────────────┐ │ │
           │                                               │ │ │  Upsample   │ │ │
           │                                               │ │ │+ Skip(feat3)│ │ │
           │                                               │ │ │ Dec Block 1 │ │ │
           │                                               │ │ │  (512 ch)   │ │ │
           │                                               │ │ └──────┬──────┘ │ │
           │                                               │ │ ┌──────▼──────┐ │ │
           │                                               │ │ │  Upsample   │ │ │
           │                                               │ │ │+ Skip(feat2)│ │ │
           │                                               │ │ │ Dec Block 2 │ │ │
           │                                               │ │ │  (256 ch)   │ │ │
           │                                               │ │ └──────┬──────┘ │ │
           │                                               │ │ ┌──────▼──────┐ │ │
           │                                               │ │ │  Upsample   │ │ │
           │                                               │ │ │+ Skip(feat1)│ │ │
           │                                               │ │ │ Dec Block 3 │ │ │
           │                                               │ │ │  (128 ch)   │ │ │
           │                                               │ │ └──────┬──────┘ │ │
           │                                               │ │ ┌──────▼──────┐ │ │
           │                                               │ │ │  Upsample   │ │ │
           │                                               │ │ │ Dec Block 4 │ │ │
           │                                               │ │ │   (64 ch)   │ │ │
           │                                               │ │ └──────┬──────┘ │ │
           │                                               │ │ ┌──────▼──────┐ │ │
           │                                               │ │ │   1×1 Conv  │ │ │
           │                                               │ │ │ 2 classes   │ │ │
           │                                               │ │ └──────┬──────┘ │ │
           │                                               │ └────────┼────────┘ │
           │                                               │          │          │
           │                                               │    Logits→Sigmoid   │
           │                                               │          │          │
           │                                               │    prob = σ(L)      │
           │                                               │    mask = (p>0.5)   │
           │                                               │                     │
           │                                               │ Speed: ~40 FPS GPU  │
           │                                               │                     │
           └───────────────────┬───────────────────────────┴─────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. MULTI-MODAL FUSION (optional)                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │ Thermal Mask │    │  HSI Mask    │    │ Fusion Logic │             │
│  │ (buildings,  │───►│ (spectral    │───►│ • Union      │             │
│  │  vehicles,   │    │  anomalies)  │    │ • Intersection│            │
│  │  industrial) │    │              │    │ • Weighted   │             │
│  └──────────────┘    └──────────────┘    └──────┬───────┘             │
│                                                   │                      │
│                                          fused_mask = M_th ∪ M_hsi      │
│                                                                          │
└──────────────────────────────────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. POST-PROCESSING (configurable via no_filters flag)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  if no_filters = false:                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │ Morphological│───►│     CRF      │───►│  Geometric   │             │
│  │  • Opening   │    │  Refinement  │    │   Filters    │             │
│  │  • Closing   │    │  (optional)  │    │ • min_area   │             │
│  │  • Remove    │    │              │    │ • aspect     │             │
│  │    speckles  │    │              │    │ • circularity│             │
│  └──────────────┘    └──────────────┘    └──────┬───────┘             │
│                                                   │                      │
│  if no_filters = true:                           │                      │
│    → Skip directly to material characterization  │                      │
│                                                   │                      │
└──────────────────────────────────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. MATERIAL CHARACTERIZATION                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ THERMAL EMISSIVITY ANALYSIS                                     │   │
│  │                                                                  │   │
│  │  • Extract anomaly pixel temperatures                           │   │
│  │  • Compute mean, max, std, local contrast                       │   │
│  │  • Heuristic classification:                                    │   │
│  │                                                                  │   │
│  │    if contrast > 2.5σ AND T > p95:                              │   │
│  │      → LOW_EMISSIVITY_METAL/ROOF                                │   │
│  │    elif contrast > 1.0σ:                                        │   │
│  │      → MEDIUM_EMISSIVITY_ASPHALT/CONCRETE                       │   │
│  │    else:                                                         │   │
│  │      → HIGH_EMISSIVITY_SOIL/VEGETATION                          │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ SPECTRAL MATERIAL ANALYSIS (Hyperspectral)                      │   │
│  │                                                                  │   │
│  │  • Extract mean spectrum from anomaly pixels                    │   │
│  │  • Compute absorption depths:                                   │   │
│  │      d_1400, d_1900 (water)                                     │   │
│  │      d_1000 (Fe-oxide)                                          │   │
│  │      d_2200 (OH/clay)                                           │   │
│  │                                                                  │   │
│  │  • Material classification:                                     │   │
│  │    if d_1400 < 0.1 AND d_1900 < 0.1:                            │   │
│  │      → DRY_MINERAL/ASPHALT/CONCRETE                             │   │
│  │    if d_1000 > 0.25 AND d_2200 > 0.2:                           │   │
│  │      → OXIDIZED_MINERAL/CLAY                                    │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. OUTPUT GENERATION                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │  GeoTIFF     │    │ Probability  │    │     RGB      │             │
│  │   Masks      │    │  Heatmaps    │    │   Overlays   │             │
│  │ (binary LZW) │    │ (Float32)    │    │ (300 DPI PNG)│             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ STRUCTURED REPORTS (JSON/CSV)                                    │  │
│  │                                                                   │  │
│  │  {                                                                │  │
│  │    "scene_id": "LC08_001002_20231015",                           │  │
│  │    "metrics": {                                                   │  │
│  │      "accuracy": 0.95, "f1_score": 0.90,                         │  │
│  │      "precision": 0.92, "recall": 0.88                           │  │
│  │    },                                                             │  │
│  │    "detections": [                                                │  │
│  │      {                                                            │  │
│  │        "id": 1,                                                   │  │
│  │        "centroid": [34.052, -118.243],                           │  │
│  │        "area_pixels": 1523,                                      │  │
│  │        "confidence": 0.94,                                       │  │
│  │        "material_class": "ASPHALT/CONCRETE",                     │  │
│  │        "mean_temp": 312.5,                                       │  │
│  │        "local_contrast": 8.3                                     │  │
│  │      }                                                            │  │
│  │    ]                                                              │  │
│  │  }                                                                │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING WORKFLOW                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ config.yaml  │
│  + splits    │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ DATA MODULE INITIALIZATION                                              │
│  • Scan data/ directory for supported formats                           │
│  • Load train.json, val.json, test.json splits                          │
│  • Create DataLoaders (batch_size=8, workers=8, pin_memory=True)        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MODEL INITIALIZATION                                                     │
│  • Create Swin U-Net (backbone=swin_tiny, input_ch=1, classes=2)        │
│  • Move to GPU, wrap in DataParallel if multi-GPU                       │
│  • Initialize weights (kaiming_normal for conv, trunc_normal for linear)│
│  • Count parameters (~28M for swin_tiny)                                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ OPTIMIZER & SCHEDULER SETUP                                             │
│  • Optimizer: AdamW (lr=1e-4, wd=1e-5, betas=(0.9,0.999))               │
│  • Scheduler: CosineAnnealingLR (T_max=epochs)                          │
│  • GradScaler: torch.amp.GradScaler for mixed precision                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TRAINING LOOP (epochs=50)                                               │
│                                                                          │
│  FOR epoch IN range(epochs):                                            │
│                                                                          │
│    ┌──────────────────────────────────────────────────────────────┐    │
│    │ TRAIN EPOCH                                                  │    │
│    │                                                               │    │
│    │  model.train()                                               │    │
│    │                                                               │    │
│    │  FOR batch IN train_loader:                                  │    │
│    │    1. Load: images, masks → GPU                              │    │
│    │    2. Forward (with AMP):                                    │    │
│    │         with autocast():                                     │    │
│    │           logits = model(images)                             │    │
│    │           L = combined_loss(logits, masks)                   │    │
│    │             = 0.5·BCE + 0.3·Dice + 0.2·Focal                 │    │
│    │    3. Backward:                                              │    │
│    │         scaler.scale(L).backward()                           │    │
│    │    4. Gradient clip:                                         │    │
│    │         scaler.unscale_(optimizer)                           │    │
│    │         clip_grad_norm_(params, max_norm=1.0)                │    │
│    │    5. Optimizer step:                                        │    │
│    │         scaler.step(optimizer)                               │    │
│    │         scaler.update()                                      │    │
│    │    6. Accumulate metrics (loss, predictions, targets)        │    │
│    │                                                               │    │
│    │  Compute epoch metrics:                                      │    │
│    │    accuracy, F1, precision, recall, ROC-AUC, PR-AUC          │    │
│    │                                                               │    │
│    └───────────────────────────────────────────────────────────────┘    │
│                                                                          │
│    ┌──────────────────────────────────────────────────────────────┐    │
│    │ VALIDATION EPOCH (every val_frequency=5 epochs)              │    │
│    │                                                               │    │
│    │  model.eval()                                                │    │
│    │                                                               │    │
│    │  with torch.no_grad():                                       │    │
│    │    FOR batch IN val_loader:                                  │    │
│    │      1. Load: images, masks → GPU                            │    │
│    │      2. Forward (with AMP):                                  │    │
│    │           logits = model(images)                             │    │
│    │           L = combined_loss(logits, masks)                   │    │
│    │      3. Accumulate metrics                                   │    │
│    │                                                               │    │
│    │  Compute epoch metrics:                                      │    │
│    │    accuracy, F1, precision, recall, ROC-AUC, PR-AUC          │    │
│    │                                                               │    │
│    └───────────────────────────────────────────────────────────────┘    │
│                                                                          │
│    ┌──────────────────────────────────────────────────────────────┐    │
│    │ CHECKPOINT & EARLY STOPPING                                  │    │
│    │                                                               │    │
│    │  if val_f1 > best_f1:                                        │    │
│    │    best_f1 = val_f1                                          │    │
│    │    save_checkpoint(epoch, model, optimizer, "best_model.pth")│    │
│    │    patience_counter = 0                                      │    │
│    │  else:                                                        │    │
│    │    patience_counter += 1                                     │    │
│    │                                                               │    │
│    │  if patience_counter >= patience (15):                       │    │
│    │    STOP TRAINING (early stopping)                            │    │
│    │                                                               │    │
│    └───────────────────────────────────────────────────────────────┘    │
│                                                                          │
│    scheduler.step()  # Update learning rate                             │
│    log_metrics_to_csv(epoch, train_metrics, val_metrics, lr)            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ FINAL EVALUATION                                                         │
│  • Load best_model.pth                                                   │
│  • Run on test set                                                       │
│  • Compute final metrics                                                 │
│  • Generate sample submission outputs                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🧮 Mathematical Framework

### 1. Robust Normalization
```
Per-band scaling:
  μ_b = median(X_b)                    [robust central tendency]
  σ_b = median(|X_b − μ_b|) + ε       [robust scale, MAD-based]
  Z_b = (X_b − μ_b) / σ_b             [normalized band]

Example:
  Raw thermal band: [295, 298, 300, 302, 315, 305, 301, 299]
  μ = 300.5, σ = 2.5
  Normalized: [-2.2, -1.0, -0.2, 0.6, 5.8, 1.8, 0.2, -0.6]
  → Outlier at 315 clearly visible (z=5.8)
```

### 2. Fast Anomaly Detection

**Statistical Thermal Thresholding:**
```
Land mask: M_land = (T > 0)
Background stats over land:
  μ_land = mean(T[M_land])
  σ_land = std(T[M_land])

Adaptive thresholds:
  τ_building = μ_land + 2.0·σ_land
  τ_vehicle = μ_land + 3.0·σ_land
  τ_industrial = μ_land + 4.0·σ_land

Detection:
  M_anomaly = (T > τ_building) ∨ (T > τ_vehicle) ∨ (T > τ_industrial)

Example (Celsius):
  μ_land = 20°C, σ_land = 3°C
  τ_building = 26°C
  τ_vehicle = 29°C
  τ_industrial = 32°C
  → Building at 28°C flagged
  → Vehicle engine at 35°C flagged
```

**RX Anomaly Detector (Hyperspectral):**
```
Robust z-scoring:
  μ = median(X, axis=pixels)           [per-band median]
  σ = median(|X − μ|, axis=pixels)     [per-band MAD]
  Z = (X − μ) ⊘ σ                      [element-wise division]

RX score (diagonal Mahalanobis):
  s_RX(x) = ||z||²₂ = Σ_b z_b²

Threshold:
  τ_RX = median(s_RX) + 6·MAD(s_RX)

Example (5 bands):
  z = [0.5, 1.2, 0.8, 6.5, 1.0]
  s_RX = 0.25 + 1.44 + 0.64 + 42.25 + 1.0 = 45.58
  If median(s_RX) = 5.0, MAD = 2.0, τ = 5+6·2 = 17
  → s_RX=45.58 > 17 → ANOMALY
```

**PCA Reconstruction Error (Generative Proxy):**
```
Fit PCA on normalized spectra:
  X_centered = X − μ
  [U, S, V] = SVD(X_centered)
  P_k = V[:, :k]                       [top k components]

Reconstruction:
  X̂_centered = P_k · P_k^T · X_centered
  X̂ = X̂_centered + μ

Reconstruction error:
  s_RE(x) = ||x_centered − x̂_centered||²₂

Combined score:
  s(x) = 0.6·normalize(s_RX(x)) + 0.4·normalize(s_RE(x))

Example:
  s_RX_norm = 0.8, s_RE_norm = 0.9
  s = 0.6·0.8 + 0.4·0.9 = 0.48 + 0.36 = 0.84
  If τ_combined = 0.7 → ANOMALY
```

### 3. Neural Network (Swin U-Net)

**Patch Embedding:**
```
Input: I ∈ R^{H×W×C}
Patch size: p = 4
Number of patches: n = (H/p) × (W/p)

Conv projection:
  E = Conv_{p×p,stride=p}(I)           [shape: (H/p, W/p, d)]
  E_flat = flatten(E)                  [shape: (n, d)]

Example:
  Input: 512×512×1
  After embedding: 128×128 patches, each 96-dim
  Total tokens: 16,384
```

**Windowed Self-Attention:**
```
Window size: w = 7
Within each w×w window:

  Q = Linear_Q(X), K = Linear_K(X), V = Linear_V(X)
  
  Attention scores:
    A = softmax((Q·K^T)/√d_head + B_rel)
  
  Output:
    Y = A·V

Relative position bias:
  B_rel[i,j] = learned_table[Δh, Δw]
  where Δh = row_i − row_j, Δw = col_i − col_j

Shifted windows (alternating blocks):
  shift_size = w/2 = 3
  X_shifted = roll(X, shifts=(-3, -3), dims=(H,W))
  → Enables cross-window interaction

Example attention:
  Patch at (5,5) attends to patches in window [(3,3) to (9,9)]
  With shift, next layer attends to window [(0,0) to (6,6)]
  → Full receptive field grows quadratically
```

**Patch Merging (Downsampling):**
```
Input: X ∈ R^{H×W×C}

Spatial downsampling by 2:
  X_0 = X[0::2, 0::2, :]               [top-left quadrant]
  X_1 = X[1::2, 0::2, :]               [bottom-left]
  X_2 = X[0::2, 1::2, :]               [top-right]
  X_3 = X[1::2, 1::2, :]               [bottom-right]
  
  X_concat = concat([X_0, X_1, X_2, X_3], dim=-1)  [shape: (H/2, W/2, 4C)]
  
  X_out = LayerNorm(X_concat)
  X_out = Linear_{4C→2C}(X_out)        [shape: (H/2, W/2, 2C)]

Example:
  Input: 128×128×96
  Output: 64×64×192
  → Halves spatial resolution, doubles channels
```

**U-Net Decoder:**
```
FOR each decoder stage i:
  
  1. Upsample by 2:
       X_up = interpolate(X, scale_factor=2, mode='bilinear')
  
  2. Concatenate skip connection:
       X_cat = concat([X_up, skip_features_i], dim=channels)
  
  3. Convolutional refinement:
       X_out = ReLU(BN(Conv_3×3(X_cat)))
       X_out = ReLU(BN(Conv_3×3(X_out)))

Example:
  Bottleneck: 16×16×768
  ↓ Upsample: 32×32×768
  ↓ + Skip(32×32×384): 32×32×1152
  ↓ Conv blocks: 32×32×512
```

**Loss Functions:**
```
Binary Cross-Entropy:
  BCE(p, y) = -[y·log(p) + (1−y)·log(1−p)]
  where p = σ(logits)

Dice Loss:
  intersection = Σ(p · y)
  union = Σp + Σy
  Dice = (2·intersection + ε) / (union + ε)
  Loss_Dice = 1 − Dice

Focal Loss (handles class imbalance):
  p_t = p if y=1 else (1−p)
  Focal = -α·(1 − p_t)^γ · log(p_t)
  α=1, γ=2 (default)

Combined:
  L_total = 0.5·BCE + 0.3·Dice + 0.2·Focal

Example:
  For hard negative (y=0, p=0.1):
    BCE = -log(0.9) = 0.105
    p_t = 0.9, (1−p_t)^2 = 0.01
    Focal = -1·0.01·log(0.9) = 0.001
  → Focal downweights easy examples
```

### 4. Material Characterization

**Thermal Emissivity Classification:**
```
Given anomaly mask M and thermal image T:

Extract anomaly temperatures:
  T_anom = T[M]
  μ_anom = mean(T_anom)
  σ_anom = std(T_anom)
  T_max = max(T_anom)

Compute local contrast:
  T_background = mean(T[¬M])
  contrast = μ_anom − T_background

Classification rules:
  IF contrast > 2.5·σ_anom AND T_max > percentile(T, 95):
    → LOW_EMISSIVITY_METAL/ROOF (ε ≈ 0.85-0.92)
  
  ELIF contrast > 1.0·σ_anom:
    → MEDIUM_EMISSIVITY_ASPHALT/CONCRETE (ε ≈ 0.93-0.96)
  
  ELSE:
    → HIGH_EMISSIVITY_SOIL/VEGETATION (ε ≈ 0.95-0.98)

Example:
  μ_anom = 315K, σ_anom = 2K, T_background = 300K
  contrast = 15K, 2.5·σ = 5K
  15 > 5 → Metal roof detected
```

**Spectral Material Classification:**
```
Extract mean spectrum from anomaly:
  S = mean(X[M, :], axis=pixels)      [average spectrum]

Normalize for depth computation:
  S_norm = (S − min(S)) / (max(S) − min(S) + ε)

Absorption depth at wavelength λ:
  d(λ) = 1 − S_norm[band_closest_to_λ]

Key features:
  d_1400 = depth at 1400nm (water)
  d_1900 = depth at 1900nm (water)
  d_1000 = depth at 1000nm (Fe-oxide)
  d_2200 = depth at 2200nm (OH/clay)

Classification:
  IF d_1400 < 0.1 AND d_1900 < 0.1:
    → DRY_MINERAL/ASPHALT/CONCRETE (low water)
  
  IF d_1000 > 0.25 AND d_2200 > 0.2:
    → OXIDIZED_MINERAL/CLAY (iron+clay signature)

Example:
  d_1400 = 0.05, d_1900 = 0.03 → Dry material (asphalt)
  d_1000 = 0.30, d_2200 = 0.25 → Oxidized clay
```

---

## 🚀 Quick Start

### Installation
```bash
# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install rasterio numpy pandas scikit-learn opencv-python tqdm pyyaml h5py scipy pillow
```

### Training
```bash
# Edit config/config.yaml first
python src/train/train.py
```

### Inference
```python
from src.inference.thermal_inference import FastManMadeInference

config = {...}  # Load from config.yaml
detector = FastManMadeInference(config)

# Single image
mask = detector.run_inference("data/scene_001_ST_B10.TIF")

# Batch processing
results = detector.batch_inference("data/")

# Material characterization
materials = detector.characterize_thermal_emissivity(thermal_data, mask)
```

---

## 📁 Project Structure

```
AIGR-S86462EUCLIDEAN_TECHNOLOGIES_THERMAL/
├── config/config.yaml           # Configuration
├── data/                        # Input data
├── src/
│   ├── dataloader/             # Data loading
│   ├── models/                 # Model architectures
│   ├── train/                  # Training pipeline
│   ├── inference/              # Inference engine
│   └── utils/                  # Utilities
├── outputs/
│   ├── models/                 # Checkpoints
│   ├── logs/                   # Training logs
│   └── submission/             # Results
└── README.md
```

---

## 📊 Performance Metrics

- **Accuracy**: Pixel-wise classification accuracy
- **F1 Score**: Harmonic mean of precision/recall
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve

---

## 🛡️ Security & Ethics

**Implemented:**
- Mixed precision training (numerical stability)
- Gradient clipping (prevents instability)
- Uncertainty quantification (flags low confidence)
- Audit logging (tracks all operations)
- Configuration validation (safe parameters)
- No-filters mode (user control)

**Ethical AI:**
- Transparent predictions with confidence scores
- Human-in-the-loop for ambiguous cases
- Bias mitigation through diverse validation
- Designed for environmental/disaster monitoring

---

## 📧 Contact

**Euclidean Technologies**  
Email: contact@euclideantech.ai  
Project: AIGR-S86462

---

**Built  by Euclidean Technologies**
