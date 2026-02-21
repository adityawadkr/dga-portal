# DGA Analysis Portal — Transformer Fault Intelligence

> Physics-Informed Neural Networks for Dissolved Gas Analysis (DGA) in power transformers. Multi-model diagnostic system with IEEE C57.104 compliance.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

### 🧠 AI Models
| Model | Type | Accuracy | Description |
|-------|------|----------|-------------|
| **PINN 7-Class** | Physics-Informed NN | 99.8% F1 | Learnable gas ratio constraints, focal loss |
| **Random Forest v2** | Ensemble | 98.6% F1 | 300-tree calibrated, 2100 synthetic samples |
| **DP PINN** | Regression (PINN) | R²=0.86 | Predicts paper insulation degree of polymerization |
| **Duval Triangle** | IEEE Standard | — | Ratio-based with TDCG threshold gate |
| **Rogers Ratio** | Classic | — | Gas ratio method |

### 📊 Diagnostic Views
- **Global Dashboard** — Fleet-wide metrics with critical/warning counts, auto-loaded from DGADATA.xlsx
- **Manual Input** — Enter 7 gas concentrations, get 5-method consensus diagnosis with infographics
- **Batch Upload** — Upload `.xlsx/.csv/.tsv/.json/.txt`, auto-detect columns, analyze up to 200 transformers
- **Trend Analysis** — 12-month gas evolution charts with Rate-of-Change (ROC) warnings
- **Model Leaderboard** — All models ranked with confusion matrices and feature importance
- **Maintenance Recommendations** — IEEE-based action items per fault type and DP health

### 🔬 Infographics (per transformer)
- Radar chart (gas profile)
- Model confidence comparison
- IEEE C57.104 threshold status cards
- Duval Triangle 1 with interactive zones
- Class probability bars
- Key gas ratios (CH₄/H₂, C₂H₂/C₂H₄, etc.)
- Maintenance recommendations with severity assessment

### 📥 Export
- **CSV** — Full fleet analysis with all model predictions
- **PDF** — Single transformer diagnostic report (A4)

## Quick Start

```bash
# Clone
git clone https://github.com/adityawadkr/dga-portal.git
cd dga-portal

# Install dependencies
pip install -r requirements.txt

# Run
cd dga_portal
python app.py
# → http://127.0.0.1:5050
```

## Project Structure

```
dga-portal/
├── dga_portal/
│   ├── app.py                  # Flask backend (v6) — all endpoints
│   ├── templates/
│   │   └── index.html          # Single-page frontend
│   ├── static/img/             # Confusion matrices, feature importance
│   └── uploads/                # Temporary upload storage
├── pinn_7class.py              # PINN 7-class model architecture
├── dp_pinn.py                  # DP PINN model architecture
├── pinn_7class_model.pt        # Trained PINN weights
├── dp_pinn_model.pt            # Trained DP PINN weights
├── dga_model_v2.joblib         # Trained RF v2 model
├── DGADATA.xlsx                # Real DGA dataset (510 samples)
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | Check loaded models |
| `POST` | `/api/predict` | Single transformer diagnosis |
| `POST` | `/api/upload` | Batch file analysis |
| `GET` | `/api/fleet` | Auto-loaded fleet data |
| `POST` | `/api/recommend` | Maintenance recommendations |
| `POST` | `/api/condition` | IEEE C57.104 condition score |
| `POST` | `/api/compare` | Compare 2-3 transformers |
| `GET/POST` | `/api/audit` | Session audit log |
| `GET` | `/api/leaderboard` | Model performance rankings |

## IEEE C57.104 Compliance

The portal implements IEEE C57.104-2019 dissolved gas thresholds:
- **TDCG Threshold (720 ppm)** — Classic methods (Duval, Rogers, IEC) return "Normal" below this
- **ML Threshold (50 ppm)** — AI models return "Normal" for extremely clean oil samples
- **4-Condition System** — Per-gas condition rating (1-4) with severity colors

## Tech Stack

- **Backend**: Flask, PyTorch, scikit-learn, pandas
- **Frontend**: Vanilla JS, Chart.js, html2pdf.js
- **Design**: Custom design system (Inter + JetBrains Mono + Newsreader fonts)
- **Models**: Physics-Informed Neural Networks with learnable gas constraints

## License

MIT License — Aditya Wadkar
