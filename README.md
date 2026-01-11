# Hybrid Clustering and Recommendation System for E-commerce

MSc Data Sciences and Business Analytics - CentraleSupélec, Université Paris-Saclay

**Team Members:** Chien-Wei WENG, Ke Chen, Nicolas Perion-Quémeneur, Zihan Yang

---

## Overview

Customer segmentation-based recommendation system for e-commerce using Instacart grocery data. Validates the hypothesis that segment-specific recommendation models outperform global models.

**Dataset:** [Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) (206K users, 49K products, 3.3M orders)

---

## Project Structure
```
├── notebooks/
│   ├── 01_phase0_data_preparation.ipynb
│   ├── 02_phase1_clustering.ipynb
│   ├── 03_phase2_recommendation.ipynb
│   └── 04_phase3_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py              # Data loading utilities
│   └── recommendation.py          # Recommendation functions
├── results/figures/                # Key visualizations
├── requirements.txt
├── DATASET.md
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Anaconda (recommended) or Python with pip
- See `requirements.txt` for full dependencies

---

<details>
<summary><b>Option 1: Using Anaconda (Recommended) - Click to expand</b></summary>

### Step 1: Create Environment
```bash
# Open Anaconda Prompt (Windows) or Terminal (Mac/Linux)
conda create -n instacart python=3.11
conda activate instacart
```

### Step 2: Install Dependencies
```bash
# Install scikit-surprise via conda (required)
conda install -c conda-forge scikit-surprise

# Install remaining packages
pip install -r requirements.txt
```

### Step 3: Launch Jupyter
```bash
# Start Jupyter Lab
jupyter lab

# OR if using VS Code:
# 1. Open project folder in VS Code
# 2. Select kernel: "Python 3.11 (instacart)"
# 3. Open notebook and run
```

### Step 4: Run Notebooks Sequentially
- Execute `01_phase0_data_preparation.ipynb` first (data downloads automatically)
- Then run `02 → 03 → 04` in order
- **Note:** Notebook 04 requires outputs from notebooks 01-03

</details>

---

<details>
<summary><b>Option 2: Using pip (Alternative) - Click to expand</b></summary>

If you don't have Anaconda:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies (Note: scikit-surprise may require C++ compiler)
pip install -r requirements.txt
```

**Note:** `scikit-surprise` installation via pip may fail on Windows without Visual Studio Build Tools. Anaconda method is strongly recommended.

</details>

---

### Expected Runtime
- **Notebook 01:** ~5-10 minutes (includes data download)
- **Notebook 02:** ~10-20 minutes
- **Notebook 03:** ~60 minutes
- **Notebook 04:** ~60-90 minutes (full evaluation)

**Total:** ~3 hours for complete execution

---

<details>
<summary><b>Troubleshooting - Click to expand</b></summary>

**Issue:** `ModuleNotFoundError: No module named 'surprise'`
- **Solution:** Install via conda: `conda install -c conda-forge scikit-surprise`

**Issue:** Notebook 04 fails with "FileNotFoundError"
- **Solution:** Run notebooks 01-03 first to generate processed data files

**Issue:** Data download fails
- **Solution:** Check internet connection, Kaggle API credentials may be required

</details>

---

## Methodology

1. **Phase 0:** Data preparation, temporal split, RFM feature engineering
2. **Phase 1:** K-means clustering (PCA: 163→99 features, k=5 segments)
3. **Phase 2:** Collaborative Filtering (SVD), Content-Based Filtering, Hybrid
4. **Phase 3:** Evaluation (Precision@K, Recall@K, F1@K on test set)

---

## Key Results

**Customer Segments Identified:**
- Power Users (45.0%) - High frequency, large baskets
- Bulk Shoppers (36.0%) - Low frequency, largest baskets
- Routine Snackers (13.4%) - Regular purchases, snack-focused
- Household Essentials (4.4%) - Utilitarian shoppers
- Alcohol Enthusiasts (1.1%) - Specialized niche

**Model Performance (Test Set, N=2000):**

| Model    | Global F1@5 | Segment F1@5 | Improvement |
|----------|-------------|--------------|-------------|
| Baseline | 0.00985     | 0.01157      | +17.5%      |
| CF       | 0.00046     | 0.00146      | +213.8%     |
| Hybrid   | 0.00157     | 0.00233      | +48.4%      |

**Finding:** Segment-specific models consistently outperform global models across all metrics (9/9 wins for CF and Hybrid), validating our hypothesis.

---

## Technical Details

- **Clustering:** K-means with PCA (80% variance), Silhouette score: 0.022
- **CF:** SVD matrix factorization, log-transformed purchase frequencies
- **CBF:** Binary feature vectors on product features (department, aisle)
- **Evaluation:** Stratified sampling (2000 users), temporal validation

---

## Contact

For questions about this project, contact the team via [university email or course platform].
