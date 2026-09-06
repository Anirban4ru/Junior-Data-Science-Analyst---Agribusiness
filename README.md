# Agribusiness Data Science & Strategy

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agribuisnessanalysis.streamlit.app/)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-pep8-green.svg)](https://peps.python.org/pep-0008/)

An end-to-end commodity data science and strategic procurement repository covering macro agro-ecological planning, biophysical telemetry pipelines, nonlinear climate shock diagnostics, and probabilistic quantile forecasting.

---

## 🚀 Live Demo & 1-Click Notebooks

| Module | 1-Click Access | Highlights |
| :--- | :--- | :--- |
| **Live Web App** | [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agribuisnessanalysis.streamlit.app/) | Interactive dashboard with district selectors, climate shock sliders & fan charts |
| **Week 1: Strategic Planning** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anirban4ru/Junior-Data-Science-Analyst---Agribusiness/blob/main/notebooks/01_week1_strategic_environmental_mapping.ipynb) | Agro-Ecological Zoning (AEZ), PESTLE quantitative model, Porter's 5 Forces, produce matrix |
| **Week 2: Preprocessing** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anirban4ru/Junior-Data-Science-Analyst---Agribusiness/blob/main/notebooks/02_week2_acquisition_preprocessing.ipynb) | Multi-modal telemetry simulator, MCAR/MAR/MNAR imputation, PCHIP, rolling IQR Winsorization |
| **Week 3: EDA & Diagnostics** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anirban4ru/Junior-Data-Science-Analyst---Agribusiness/blob/main/notebooks/03_week3_eda_visualizations.ipynb) | STL decomposition, 4 specialized domain charts, dynamic freight rerouting matrices |
| **Week 4: ML & Governance** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anirban4ru/Junior-Data-Science-Analyst---Agribusiness/blob/main/notebooks/04_week4_strategy_evaluation_models.ipynb) | ARIMA/OLS audit, GDD & Tetens VPD, Quantile Boosting ($P_{10}, P_{50}, P_{90}$), KS drift |

---

## 📂 Repository Structure

```
Junior-Data-Science-Analyst---Agribusiness/
├── app.py                                        # Interactive Streamlit dashboard
├── notebooks/
│   ├── 01_week1_strategic_environmental_mapping.ipynb
│   ├── 02_week2_acquisition_preprocessing.ipynb
│   ├── 03_week3_eda_visualizations.ipynb
│   └── 04_week4_strategy_evaluation_models.ipynb
├── data/
│   ├── clean_agritech_telemetry.parquet          # High-performance Parquet format (173 KB)
│   └── clean_agritech_telemetry.csv              # CSV fallback format (410 KB)
├── requirements.txt                              # Pinned dependencies
└── pyproject.toml                                # Packaging configuration
```

---

## ⚡ Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/Anirban4ru/Junior-Data-Science-Analyst---Agribusiness.git
cd Junior-Data-Science-Analyst---Agribusiness

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Locally
- **Launch Interactive Web App**:
  ```bash
  streamlit run app.py
  ```
- **Launch Jupyter Notebooks**:
  ```bash
  jupyter lab notebooks/
  ```

---
*All notebooks are self-contained and execute sequentially with zero external dependencies.*
