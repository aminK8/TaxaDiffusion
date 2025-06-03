# 🐟 <h1 align="center">
  <span style="color:#e6194B;">T</span>
  <span style="color:#f58231;">a</span>
  <span style="color:#ffe119;">x</span>
  <span style="color:#bfef45;">a</span>
  <span style="color:#3cb44b;">D</span>
  <span style="color:#42d4f4;">i</span>
  <span style="color:#4363d8;">f</span>
  <span style="color:#911eb4;">f</span>
  <span style="color:#f032e6;">u</span>
  <span style="color:#a9a9a9;">s</span>
  <span style="color:#fabed4;">i</span>
  <span style="color:#dcbeff;">o</span>
  <span style="color:#469990;">n</span>
</h1>: Progressively Trained Diffusion Model for Fine-Grained Species Generation

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

- **Amin Karimi Monsefi** – The Ohio State University  
- **Mridul Khurana** – Virginia Tech  
- **Rajiv Ramnath** – The Ohio State University  
- **Anuj Karpatne** – Virginia Tech  
- **Wei-Lun Chao** – The Ohio State University  
- **Cheng Zhang** – Texas A&M University

---

## 📦 Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/aminK8/TaxaDiffusion.git
cd TaxaDiffusion

# Create virtual environment
python3 -m venv taxa_env
source taxa_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```
