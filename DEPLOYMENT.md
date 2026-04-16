# 🚀 Deployment Guide: Elite Estates

This guide details how to deploy and run the **Elite Estates** House Price Prediction system. The application consists of a high-performance FastAPI backend and a premium, glassmorphism-inspired web frontend.

## 🏗️ System Architecture
- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML5/CSS3 with Lucide Icons (Premium Design)
- **Monitoring**: Prometheus (Metrics) & Grafana (Visual Dashboards)
- **Deployment**: Docker & Docker Compose

---

## 💻 Local Deployment (Windows)

To run the application locally without Docker, follow these steps:

### 1. Setup Environment
Ensure you have Python 3.9+ installed and install the dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Start the API Server
Run the following command from the project root:
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```
- **Web UI**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## 🐳 Containerized Deployment (Recommended)

For a production-ready setup with full monitoring, use Docker Compose.

### 1. Launch Services
Run the following command to build and start the API, Prometheus, and Grafana:
```powershell
docker-compose up --build -d
```

### 2. Access Points
| Service | URL | Note |
| :--- | :--- | :--- |
| **Elite Estates UI** | [http://localhost:8000](http://localhost:8000) | Main Prediction Interface |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Raw Metrics & Queries |
| **Grafana** | [http://localhost:3000](http://localhost:3000) | Visual Performance Dashboards |

### 3. Monitoring Login
- **User**: `admin`
- **Password**: `admin` (can be changed in `docker-compose.yml`)

---

## 🎨 UI Features
The new **Elite Estates** interface includes:
- **Glassmorphism Design**: Modern, semi-transparent layouts with backdrop blur.
- **Dynamic Background**: High-resolution architectural imagery.
- **Micro-animations**: Smooth loading states and result transitions.
- **Real-time Health Check**: Visual indicator of system status.

---

## 🛠️ Troubleshooting
- **Model not found**: Ensure `artifacts/model.joblib` exists. Run `python train.py` if missing.
- **Port Conflict**: If port 8000, 9090, or 3000 is occupied, update the mappings in `docker-compose.yml`.
- **Docker Errors**: Ensure Docker Desktop is running and WSL2 backend is enabled.

> [!TIP]
> Use the `/feedback` endpoint to send ground-truth prices back to the system. This allows Prometheus to calculate real-time accuracy metrics (MAPE, RMSE) which can be viewed in Grafana.

---

## Vercel Deployment (Public API)

This repo includes a Vercel entrypoint at `api/index.py` and a `vercel.json` that routes all paths to the FastAPI app.

### 1. Ensure model artifacts exist
Run training once so Vercel has the model in the repo (or upload them through another artifact flow):
```powershell
python train.py
```
This produces:
- `artifacts/model.joblib`
- `artifacts/metadata.json` (includes baseline metrics + reference distributions for drift)

### 2. Deploy
From the project root:
```powershell
npx vercel deploy --prod
```

### 3. Endpoints to test
- `/` (UI)
- `/docs` (Swagger)
- `/health`
- `/model/info`
- `/predict`
- `/drift/report`

### Drift + accuracy on drifted data
Call `/drift/report` with a batch. If you include `actual_price`, the response includes batch MAE/RMSE/MAPE plus deltas vs the training baseline.
