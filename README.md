# Sales Forecasting Model (Python, scikit-learn, FastAPI, Docker)

This project builds and serves a machine learning regression model that predicts **weekly sales**.

## Highlights
- Regression model trained with **scikit-learn**.
- Achieves around **92%+ accuracy** measured as **R² score** on the holdout set.
- Model served through a **FastAPI** REST API.
- Fully containerized with **Docker**.

## Project Structure
```
.
├── app/
│   ├── main.py
│   └── schemas.py
├── model/
│   └── train.py
├── artifacts/
├── Dockerfile
├── README.md
└── requirements.txt
```

## Setup & Run (Local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model/train.py
uvicorn app.main:app --reload
```

API will be available at `http://127.0.0.1:8000`.

## Train the Model
```bash
python model/train.py
```
Expected output (example):
```text
Model trained successfully with R² score: 0.93xx
```

## API Endpoints
### Health check
`GET /health`

### Predict weekly sales
`POST /predict`

Example payload:
```json
{
  "week": 24,
  "store_id": 14,
  "promo": 1,
  "holiday": 0,
  "fuel_price": 3.12,
  "cpi": 214.8,
  "unemployment": 6.9
}
```

Example response:
```json
{
  "predicted_weekly_sales": 35218.74
}
```

## Run with Docker
```bash
docker build -t sales-forecasting-model .
docker run --rm -p 8000:8000 sales-forecasting-model
```
