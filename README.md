# Vision-Language Model for Flowers Classification
<img width="913" height="327" alt="c2dfeed3ffe67da5a5a7d19a74ca873a58b55449" src="https://github.com/user-attachments/assets/4aaca7ba-d8fb-442a-9ce4-b951e731dfe3" />

## 🎯 Challenge Objective
The goal of this project is to analyze how different fine-tuning strategies affect the performance of pretrained multimodal models. Specifically, we investigate whether performance gains in zero-shot classification stem from **fine-tuning pretrained representations**, the **choice of encoder architecture**, or **hyperparameter ablations**.

We explore the transition from contrastive learning (CLIP) to generative multimodal models (**Qwen2-VL**) to understand the impact of model scale and training complexity on classification accuracy.

---

## 📊 Dataset & Data Split
We use the [Oxford Flowers 102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), consisting of 102 flower categories common in the UK.

* **Images**: 102 categories.
* **Labels & Splits**: Using the official `labels.npz` and `cat_to_name.json` for category mapping.
* **Custom Split Logic**: In accordance with the challenge requirements, we swapped the standard splits:
    * **Training Set**: Original `valid` + `tstid` indices.
    * **Test Set**: Original `trnid` (training) indices.

---

## 🛠️ Project Structure

### 1. CLIP Fine-Tuning (Contrastive)
* `train.py`: Main script for fine-tuning CLIP using a contrastive loss (InfoNCE) with AdamW.
* `baseline_eval.py`: Evaluates a vanilla, pretrained CLIP model (zero-shot) to establish a performance floor.
* `test_models.py`: Utility script to compare parameter counts across CLIP architectures (e.g., ViT-B/32 vs ViT-L/14).

### 2. Dataset Exploration & Visualization (`data_Exploration/`)
* `dataset_Description.py`: Generates a bar chart showing the class distribution across the 102 categories and prints image counts per split.
* `class_distribution.py`: Computes and plots the Kernel Density Estimation (KDE) of **Hue, Saturation, and Value (HSV)** for a specific flower class.
* `isomap.py`: Performs **Isomap dimensionality reduction** on class centroids (HSV features) and visualizes the manifold using representative images.


### 4. Interactive Tools
* `web.py`: A **Streamlit-based** web application to explore the dataset, filter by class name, and view images in grid or single-focus modes.

---

## 🚀 Key Experiments

### A. Architecture Ablation
We compare different CLIP backbones to see if larger vision encoders (e.g., ViT-L) outperform smaller ones even without extensive fine-tuning.
* **Command**: `python test_models.py`

### B. Fine-Tuning vs. Zero-Shot
Establishing the baseline with vanilla CLIP vs. fine-tuning on the specialized prompt: *"an image of the {} flower"*.
* **Baseline**: `python baseline_eval.py --architecture ViT-B/32`
* **Fine-tuned**: `python train.py --lr 5e-6 --epochs 5 --architecture ViT-B/32`

---

## 📚 Theoretical Component: InfoNCE Loss

The training objective is based on the **InfoNCE loss**, which aligns image and text representations:

* **Alignment**: The loss maximizes the similarity of the $N$ matching pairs in a batch while minimizing the similarity of the $N^2 - N$ incorrect pairs.
* **Temperature ($\tau$)**: A scalar scaling factor that controls the sharpness of the similarity distribution. Lower $\tau$ values force the model to concentrate on the hardest negative samples.

---

## 📈 Monitoring & Usage

1. **Monitor with WandB**: All runs track Top-1 accuracy and Top-3/5 retrieval.
2. **Launch Web Explorer**:
   ```bash
   streamlit run web.py
3. **Run Distribution Analysis**:
   ```bash
   python dataset_exploration/dataset_Description.py
