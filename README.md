# Job Recommender – Streamlit App

This repository contains **job_recommender_app_v7.py**, a Streamlit web application that suggests job postings based on a trained LightGBM model and user-defined filters.

## Project Structure

```
├── job_recommender_app_v7.py               # Streamlit front-end
├── job_apply_lgbm_pipeline.pkl             # Trained LightGBM pipeline (≈1.8 MB)
├── final_dataset_ml_ready_numeric_plus_extended_with_title.csv  # Feature dataset (≈9 MB)
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Local Usage

```bash
# 1. Install deps (inside venv)
pip install -r requirements.txt

# 2. Run the application
streamlit run job_recommender_app_v7.py
```

Open the link printed by Streamlit (default: http://localhost:8501) in your browser or phone on the same network.

## Deployment – Streamlit Community Cloud

1. Push this repository to GitHub.
2. Log in to https://streamlit.io/cloud and click **"New app"**.
3. Select the repo & branch, set **Main file** to `job_recommender_app_v7.py`.
4. Click **Deploy**. First build takes 1-2 minutes.

The app will be publicly available at `https://<your-app-name>-<your-user>.streamlit.app` and is responsive on mobile devices.

## Alternative Deployment (Docker)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "job_recommender_app_v7.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

Build & run:
```bash
docker build -t job-recommender .
docker run -p 8501:8501 job-recommender
```

---
**Authors** – Your Name Here 