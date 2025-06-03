# 🐟 TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation

[![Project Page](https://img.shields.io/badge/Webpage-🖼️%20Project%20Page-blue)](https://amink8.github.io/TaxaDiffusion/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01923-b31b1b.svg)](https://arxiv.org/pdf/2506.01923)

---

## ✨ Highlights

**TaxaDiffusion** introduces a progressive training strategy for fine-grained species generation using diffusion models. We incorporate taxonomic hierarchies to guide generation and improve fidelity at each biological level (Order → Family → Genus → Species).

**Key contributions:**
- Progressive taxonomy-aware generation.
- Stage-wise diffusion training from coarse to fine labels.
- Evaluation on **FishNet**, **BIOSCAN-1M**, and **iNaturalist** datasets.
- Outperforms SOTA baselines in both image quality (FID, LPIPS) and text-image alignment (BioCLIP).

![overview](images/model_overview.jpg)
*Progressive training from high-level taxonomy to fine-grained species generation.*

---

## 🧑‍💻 Authors

**Amin Karimi Monsefi**, **Mridul Khurana**, **Rajiv Ramnath**, **Anuj Karpatne**, **Wei-Lun Chao**, **Cheng Zhang**


---

## 📦 Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/aminK8/TaxaDiffusion.git
cd TaxaDiffusion

conda env create -f environment.yml
conda activate taxa_diffusion
```



## Citation
If you liked our paper, please consider citing it
```bibtex
@misc{monsefi2025taxadiffusionprogressivelytraineddiffusion,
      title={TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation}, 
      author={Amin Karimi Monsefi and Mridul Khurana and Rajiv Ramnath and Anuj Karpatne and Wei-Lun Chao and Cheng Zhang},
      year={2025},
      eprint={2506.01923},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.01923}, 
}
```
