# QARAD

**Quality‑Aware Language‑Conditioned Local Auto‑Regressive Anomaly Synthesis and Detection**

[![arXiv](https://img.shields.io/badge/arXiv-2508.03539-b31b1b.svg)](https://arxiv.org/abs/2508.03539)
![AAAI](https://img.shields.io/badge/AAAI-2026%20Poster-blue)

> **QARAD** couples a *language‑conditioned, mask‑local autoregressive editor* with a *quality‑aware re‑weighting scheme* to synthesize realistic, precisely located anomalies and train stronger anomaly detectors.

---

## 🔍 What is QARAD?

**QARAD** is a two‑component framework for industrial anomaly detection:

1. **ARAS — Auto‑Regressive Anomaly Synthesis**
   A training‑free, language‑guided, **mask‑local** editor that injects fine‑grained defects only where you ask, while **freezing the surrounding context** to preserve micro‑structure and material continuity.

2. **QAW — Quality‑Aware Weighting**
   A simple, detector‑agnostic re‑weighting that **amplifies high‑consistency synthetic samples** (measured via image–text alignment) and **down‑weights low‑consistency ones**, stabilizing optimization and improving generalization.

Together, these form **QARAD**, a synthesis‑plus‑training pipeline that delivers **controllable, realistic defects** and **robust, accurate detectors** across standard benchmarks.

**Paper:** “*Quality‑Aware Language‑Conditioned Local Auto‑Regressive Anomaly Synthesis and Detection*”
**arXiv:** [https://arxiv.org/abs/2508.03539](https://arxiv.org/abs/2508.03539)

---

## ✨ Key Contributions

* **Mask‑Local, Language‑Conditioned Editing (ARAS).**
  We introduce a hard‑gated autoregressive operator over VQ latents that *freezes* all tokens outside a user‑provided mask and **samples only within the mask**, conditioned on a natural‑language prompt. This guarantees **exact locality** and **context invariance**, enabling precise, text‑guided defect placement with sub‑pixel fidelity.

* **Quality‑Aware Re‑Weighting (QAW).**
  We compute an image–text similarity per synthetic sample and convert it into a **continuous weight** for the detector’s loss. High‑consistency syntheses receive larger gradients; low‑consistency ones are softly attenuated—**reducing gradient variance** while preserving diversity.

* **Decoupled, Plug‑and‑Play Design.**
  ARAS is **training‑free** and can be dropped into existing AD pipelines; QAW is **detector‑agnostic** and only changes training weights, not model architectures.

* **Strong Accuracy & Efficiency.**
  Across **MVTec AD**, **VisA**, and **BTAD**, QARAD delivers **consistent gains at both image‑ and pixel‑level**, while offering a **significant speed advantage** over diffusion‑based anomaly synthesis.


### ARAS (Auto‑Regressive Anomaly Synthesis)

* **Token‑anchored masked sampling.** A *hard‑gate* keeps all context tokens intact; only **masked tokens** are resampled, conditioned on the prompt.
* **Language control.** Prompts specify *type/shape/size/color/position* of the defect; small edits to the prompt yield smooth variations.
* **Micro‑structure fidelity.** Because context tokens are frozen, the synthesized region **inherits high‑frequency material statistics** (grain, weave, gloss) from its surroundings—no seam artifacts.

### QAW (Quality‑Aware Weighting)

* **Per‑sample reliability.** Compute an image–text similarity and map it through a monotone calibration (e.g., softmax) to obtain **weights**.
* **Variance reduction.** High‑quality syntheses dominate the gradient; low‑quality outliers are softly down‑weighted—**stabilizing training** without discarding data.
* **Drop‑in upgrade.** Works with standard detectors and training loops.

---

## 📊 Benchmarks (high‑level)

* Evaluated on **MVTec AD**, **VisA**, and **BTAD**.
* Demonstrates **consistent improvements** at **image‑level** and **pixel‑level** detection compared to augmentation‑based and diffusion‑based synthesis pipelines.
* **Efficiency:** ARAS avoids iterative denoising, delivering **substantial speed gains** in synthesis while keeping detector inference unchanged.

> Please see the paper for full quantitative tables, ablations, and qualitative visualizations.

---

## 🧩 Why It Works

* **Exact Locality + Context Preservation:** By editing *only* masked tokens and freezing context, ARAS eliminates low‑res bottlenecks and boundary seams that often mislead detectors.
* **Semantic Faithfulness:** Language conditioning provides **continuous** control over defect attributes beyond coarse categories.
* **Optimization with Signal, Not Noise:** QAW focuses learning on **prompt‑consistent** synthetic samples, improving robustness and generalization.

---

## ✏️ Citation

If you find QARAD useful for your research, please cite:

```bibtex
@misc{qian2025qualityawarelanguageconditionedlocalautoregressive,
      title={Quality-Aware Language-Conditioned Local Auto-Regressive Anomaly Synthesis and Detection}, 
      author={Long Qian and Bingke Zhu and Yingying Chen and Ming Tang and Jinqiao Wang},
      year={2025},
      eprint={2508.03539},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.03539}, 
}
```

---

## 📬 Contact

For questions or collaborations, please open an issue on the repository or contact the authors listed in the paper.
