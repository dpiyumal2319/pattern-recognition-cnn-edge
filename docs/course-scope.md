# Course scope index

Topic → deck → page → formula, for the EN3150 decks in `docs/slides/`.

**Why this exists.** The slides stop-gate in `AGENTS.md` points at `CNNs.pdf` (59 pages) and
`DNNs.pdf` (110 pages). Their text layer mangles every equation — `𝐹2𝐾𝐻𝑊` extracts as glyph
soup — so four issues would each pay that cost and get a poor result. Everything below was read
from the decks and retyped by hand.

**This does not replace the gate.** The gate still requires the PDF to be present. This makes
reading it fast, and keeps four people writing the same notation as the lecturer.

Every entry cites a page. If one looks wrong, the PDF wins — open an issue.

---

## 1. Convolution arithmetic

### Output spatial size — `CNNs.pdf` p32

```
input  W x W ,  filter  K x K ,  padding P

no padding :  output = W - K + 1
padding P  :  output = W - K + 1 + 2P
same size  :  set P = (K - 1) / 2
```

Stride is covered visually on p21–p24: with padding and stride 1 the spatial resolution is
unchanged; stride `S` reduces it by a factor of `1/S`.

*Used by:* #9, #10 — every layer's output shape. #11 — the derivation tables.

### Layer tensor shapes — `CNNs.pdf` p26

```
input    N x C_in  x H  x W
filters  C_out x C_in x K_h x K_w
bias     C_out
output   N x C_out x H' x W'
```

### Multiplication count — `CNNs.pdf` p27

For a padded convolution where input and output are both `H x W`:

```
multiplications = C_out x C_in x K_h x K_w x H x W

square case (C_out = C_in = C, K_h = K_w = K):  C² K² H W
```

This is the lecturer's own MAC convention and the one `src/pr_edge/profile.py` implements. Bias
additions are not counted.

*Used by:* #5 (the profiler), #11 (hand derivation). These two must agree.

### Parameter count

The deck does not state a general formula; it derives it by example on p5–p6, where a 1D
convolution with a 3-tap filter and a bias has **4 parameters (w1, w2, w3, b1)**, noted as
"usually significantly less than fully connected layer".

Generalising in that same style:

```
conv layer   :  (K_h x K_w x C_in + 1) x C_out
dense layer  :  (in_features + 1) x out_features
```

*Used by:* #11. Cross-check every hand-derived total against `results/*/profile.json`.

### Receptive field — `CNNs.pdf` p46–p51

```
stride 1 , L layers of F x F :  1 + L(F - 1)      "each successive convolution adds F - 1"
stride 2 , L layers          :  2^(L+1) - 1        grows exponentially
```

*Used by:* the architecture justification in #11 — at 64x64 input, how much of the image a
late-layer unit actually sees.

---

## 2. Pooling — `CNNs.pdf` p33–p37

Max pooling with an `F x F` window and stride `S`. **Usually F = 2 or 3, S = 2** (p34). Worked
2x2 / stride-2 example on p35–p37.

Pooling has **no learnable parameters**, which is why it does not appear in the parameter tables.
p33 gives the canonical block ordering: convolution → non-linearity → spatial pooling.

*Used by:* #9 (Model A interleaves conv and max-pool, as the brief's figure 1 shows).

---

## 3. 1x1 convolutions and dimension reduction — `CNNs.pdf` p39–p42

A `1 x 1 x K` filter, `L` of them, mixes across channels without touching spatial extent —
"network in network" (Lin et al., ICLR 2014), p41.

p42, why it is worth doing:

```
Option 1  3x3 conv, 256 -> 256          256 x 256 x 3 x 3  ~= 600,000 weights and operations
Option 2  bottleneck
          1x1 conv, 256 -> 64           256 x  64 x 1 x 1  ~=  16,000
          3x3 conv,  64 -> 64            64 x  64 x 3 x 3  ~=  36,000
          1x1 conv,  64 -> 256           64 x 256 x 1 x 1  ~=  16,000
          total                                            ~=  70,000
```

---

## 4. Depthwise separable convolutions — `CNNs.pdf` p43–p45

**This is the central idea of the whole assignment.**

p43, depthwise convolution: for an input with `K` channels, `K` separate `F x F` filters, one per
channel, costing

```
F² K H W        instead of        F² K² H W
```

followed by a pointwise `1 x 1` convolution to mix across channels.

p44, the MobileNet worked example — **reproduce this structure in #11**:

```
Regular 3x3 conv layer, 256 -> 256 channels
    9 x 256 x 256                    ~= 590,000 operations

Depthwise separable block
    step 1  3x3 depthwise, per input channel
            9 x 256                  ~=   2,300 operations  (per location)
    step 2  1x1 pointwise, in x out
            256 x 256                ~=  65,500 operations
    total                            ~=  67,800 operations

Speedup factor: 8.7
```

> **Note the deck's own rounding.** p42 gives the same `3x3 256 -> 256` layer as `~600,000` and
> p44 as `~590,000`. Exactly it is `9 x 256 x 256 = 589,824`. Likewise `9 x 256 = 2,304` and
> `256 x 256 = 65,536`, totalling `67,840`, and `589,824 / 67,840 = 8.69`. The slides round; your
> derivation should state exact figures and note that they match the lecturer's to rounding.
> Nobody has made a mistake here.

p45 generalises to group-wise convolutions: split `K` feature maps into `G` groups and convolve
within each. A depthwise convolution is the special case `G = C_in`, which is exactly
`nn.Conv2d(..., groups=in_channels)`.

Reference: A. Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile
Vision Applications*, arXiv 2017.

*Used by:* #10 (Model B), #11 (derivation), #5 (profiler verification test), #25 (task 6).

---

## 5. Where cost actually sits — `CNNs.pdf` p54

Three claims worth quoting in the report, because they frame the entire accuracy/memory/compute
trade-off:

- The majority of **parameters** reside in the fully-connected dense layers.
- The **convolution** layers account for the majority of floating-point operations.
- Most of the **memory** is used by convolution layers.

This is precisely why replacing Model A's `Flatten -> Dense` head with global average pooling is
the lever that gets Model B under 100k parameters, while the depthwise substitution is the lever
that cuts operations.

*Used by:* #10, #11, #25.

---

## 6. Transfer learning — `CNNs.pdf` p57–p58

- **Strategy 1 (p57)** — use the pretrained network as a *feature extractor*: remove the final
  layers, use the activations off the shelf.
- **Strategy 2 (p58)** — *transfer learning*: train new prediction layer(s), and either keep the
  backbone **frozen** or **fine-tune** it.

p58 links the Keras EfficientNet fine-tuning tutorial, which the assignment brief also cites.

*Used by:* #19, #20, #21.

---

## 7. Activation functions — `DNNs.pdf` p95–p99

| Page | Point |
|---|---|
| p95 | Sigmoid saturates. `σ'(x) = σ(x)(1 − σ(x))`, so `σ'(x) ≈ 0` at either extreme — the vanishing gradient problem |
| p96 | With sigmoid or tanh, be careful with weight initialisation to avoid saturation |
| p97 | **ReLU** does not saturate for positive values, and is "computationally efficient (thresholding)" |
| p98 | Dying ReLUs: neurons stuck at zero output, no gradient update. Not zero-centred. **"Recommended for most cases"** |
| p99 | Leaky ReLU and PReLU: do not saturate, neurons do not die, computationally efficient. PReLU's `α` is learnable |

**p97's "thresholding" is the hook for the hardware argument in #12.** ReLU is a single
comparison — no `exp`, no transcendental unit, no lookup table. Sigmoid, tanh and softmax all
require `exp`, which on a microcontroller without an FPU costs tens to hundreds of cycles.

*Used by:* #12 (the activation justification, part of task 2's 20 marks), #9, #10.

---

## 8. Optimizers — `DNNs.pdf` p100–p108

p100 lists what the course covers: SGD, SGD with momentum, Adam, RMSprop, Adagrad, Adadelta.

### SGD — p101

```
w  <-  w - α ∂L(w)/∂w
```

p101's stated weaknesses: can get stuck in local minima, slow to converge, saddle points.
p102 adds the conditioning problem, quoting Murphy: slow "when some directions of space point
steeply downhill, whereas other have a shallower gradient".

### SGD with momentum — p103

```
v  <-  m v - α ∂L(w)/∂w
w  <-  w + v
```

**Typically start m = 0.5 and increase** (p103). The slide's diagram decomposes the update into a
gradient step plus a momentum step giving the actual step — worth mirroring in #14's explanation
of why momentum damps oscillation across a ravine while accumulating along a consistent descent
direction.

### RMSprop — p106

```
v(k)  <-  β v(k) + (1 − β) (∂L/∂w(k))²
w(k)  <-  w(k) − α / (√v(k) + ε) · ∂L/∂w(k)
```

Decay factor `β ∈ [0, 1]`, typically `≥ 0.9`. Focuses on recent gradients, forgetting old ones.

### Adam — p107

Described as **Adagrad + momentum**:

```
m     <-  β₁ m + (1 − β₁) ∇L
v(k)  <-  β₂ v(k) + (1 − β₂) (∂L/∂w(k))²
w(k)  <-  w(k) − α / (√v(k) + ε) · m(k)
```

Hyperparameters `β₁`, `β₂`.

### Which to use — p108

The lecturer's own framing, useful for justifying the choice in task 3:

- Adaptive methods reach lower training error faster than SGD early on, and are more stable.
- Karpathy, quoted on the slide: Adam at `3e-4` as a baseline; "For ConvNets a well-tuned SGD will
  almost always slightly outperform Adam, but the optimal learning rate region is much more narrow
  and problem-specific."
- A common approach: start with Adam for 5–10 epochs, then switch to SGD.

*Used by:* #13 (the three-way comparison), #14 (momentum ablation). Task 3, 15 marks.

---

## 9. Evaluation metrics — `Learning from data.pdf` p31–p36

### Confusion matrix layout — p31, p32

The deck lays it out with **true class as rows and predicted class as columns**:

```
                  Predicted
                  Positive      Negative
True  Positive    TP            FN  (type II, miss)
      Negative    FP  (type I)  TN
```

**This matches the `rows=true, cols=predicted` convention written into every `metrics.json`.**
Keep it — a report that transposes the lecturer's own layout invites a correction.

### Formulas — p36

```
Accuracy   = (TP + TN) / (TP + TN + FP + FN)
Precision  = TP / (TP + FP)        proportion of positive predictions that are correct
Recall     = TP / (TP + FN)        also True Positive Rate, sensitivity
Specificity (TNR) = TN / (TN + FP)
FPR        = FP / (TN + FP)
F1         = 2 / (1/Precision + 1/Recall)
```

Note the deck writes F1 in **harmonic-mean reciprocal form**, not as `2PR/(P+R)`. Same quantity;
use the lecturer's form.

p31 warns that **accuracy can be misleading for imbalanced datasets**, and p33–p35 work three
numeric examples making the point. EuroSAT is close to balanced, but this is the justification for
reporting macro precision and recall rather than accuracy alone.

*Used by:* #4, #17, #18.

---

## 10. Train / validation / test methodology — `Learning from data.pdf`

| Page | Point |
|---|---|
| p15 | Overfitting, underfitting, generalisation |
| p29 | The supervised pipeline: data preparation → splitting into training / validation / test → training → validation → testing |
| p30 | Loss functions measure how far predictions are from the correct answers |
| p39 | Model selection: choose among candidates using performance metrics, tuning hyperparameters on held-out data |
| p41–p43 | k-fold cross-validation and grid search |
| p55–p57 | Bias–variance trade-off: simple model = high bias, low variance; complex model = low bias, high variance |

The 70/15/15 split in #2 follows p29. **Hyperparameters and checkpoints are selected on the
validation split; the test split is touched once.** p57's training-versus-testing error curve is
the figure to reference when discussing whether Model A overfits relative to Model B.

*Used by:* #2, #3, #6, #24.

---

## 11. Softmax — `Classification 2.pdf` p4–p5, p9

```
p(y_i = C_k | x_i, W) = exp(w_k^T x_i) / Σ_j exp(w_j^T x_i)
```

p9 gives the Jacobian, `∂ỹ_k/∂a_j = ỹ_k (𝕀(k = j) − ỹ_j)`.

Relevant here for one reason: **softmax needs `exp`, once per class per inference.** That is the
exception #12 must address in the activation argument — but it runs once over 10 values at the
output, not once per activation over a whole feature map.

---

## Quick lookup

| Need | Deck | Page |
|---|---|---|
| Output size with padding / stride | CNNs | 32, 21–24 |
| Conv multiplication count | CNNs | 27 |
| Layer tensor shapes | CNNs | 26 |
| Max pooling | CNNs | 34–37 |
| 1x1 convolutions, bottleneck | CNNs | 39–42 |
| **Depthwise separable, MobileNet example** | CNNs | 43–44 |
| Group-wise convolutions | CNNs | 45 |
| Receptive field | CNNs | 46–51 |
| Where parameters vs FLOPs sit | CNNs | 54 |
| Transfer learning, freeze vs fine-tune | CNNs | 57–58 |
| ReLU, dying ReLU, Leaky ReLU | DNNs | 97–99 |
| SGD | DNNs | 101 |
| SGD + momentum | DNNs | 103 |
| RMSprop | DNNs | 106 |
| Adam | DNNs | 107 |
| Which optimizer | DNNs | 108 |
| Confusion matrix layout | Learning from data | 31–32 |
| Accuracy / precision / recall / F1 | Learning from data | 36 |
| Train/val/test pipeline | Learning from data | 29 |
| Bias–variance | Learning from data | 55–57 |
| Softmax | Classification 2 | 4–5, 9 |
