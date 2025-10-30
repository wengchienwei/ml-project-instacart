# ML Project: Instacart Customer Segmentation & Recommendation

Hybrid clustering and recommendation system for e-commerce customer personalization using the Instacart dataset.

## Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/wengchienwei/ml-project-instacart.git
cd ml-project-instacart
```

2. **Set up environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. **Download dataset:**
```bash
python download_data.py
```

4. **Start exploring:**
Open `notebooks/01_data_exploration.ipynb`

## Team Members
- Chien-Wei Weng
- Ke Chen  
- Nicolas Perion-Quémeneur
- Zihan Yang

## Project Structure
- `notebooks/` - Jupyter notebooks for analysis
- `src/` - Python modules for reusable code
- `results/` - Generated plots and metrics
- `download_data.py` - Dataset download script


## **Check .gitignore:**
Ensure these lines are in your '.gitignore' file:
```
# Virtual environment
venv/

# Data and results (too large for git)
results/
.cache/

# Jupyter
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc
.env
```