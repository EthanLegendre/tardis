# TARDIS — TGV Delay Prediction

TARDIS is a machine learning project that predicts TGV arrival delays based on SNCF historical data. It includes an exploratory data analysis notebook, a trained Random Forest model, and an interactive Streamlit dashboard.

## Project Structure

```
.
├── tardis_eda.ipynb
├── tardis_model.ipynb
├── tardis_dashboard.py
├── model.joblib
├── cleaned_dataset.csv
├── requirements.txt
├── dataset.csv
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Launch the dashboard:

```bash
streamlit run tardis_dashboard.py
```

To run the notebooks:

```bash
jupyter notebook
```

Open `tardis_eda.ipynb` for the data analysis, then `tardis_model.ipynb` for the model training.

## Dashboard

The dashboard has three tabs:

- **Overview** — Delay distribution, delay by month, and correlation heatmap
- **Stations** — Top departure and arrival stations ranked by average delay
- **Predict** — Input a route and get a predicted arrival delay from the trained model

## Model

- Algorithm: Ridge regressor (scikit-learn)
- Target: Average delay of all trains at arrival

## Authors

- Ethan Legendre / Tristan Delvoye / Lino Iannacone — PGE1 2025/2026