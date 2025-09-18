# ❤️ Heart Disease Prediction Project

This project predicts the likelihood of heart disease using machine learning.

## 📂 Project Structure
- `data/` → dataset (`heart_disease.csv`)
- `notebooks/` → Jupyter notebooks (EDA, PCA, feature selection, supervised & unsupervised learning, tuning)
- `src/` → reusable Python scripts
- `models/` → trained ML pipeline (`final_pipeline.pkl`)
- `ui/` → Streamlit app (`app.py`)
- `results/` → evaluation metrics and plots
- `requirements.txt` → dependencies

## ⚙️ Setup
```bash
conda create -n heart_ml python=3.10 -y
conda activate heart_ml
pip install -r requirements.txt
