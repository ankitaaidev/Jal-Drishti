# Jal-Drishti

**Satellite-driven irrigation intelligence for precision agriculture.**

Jal-Drishti combines **Sentinel-1/2 satellite imagery**, **weather observations**, and **agronomic models** to estimate crop type, growth stage, moisture stress, water deficit, and irrigation priority for agricultural fields.

```
# Register a field
POST /fields

# Run inference
POST /infer/{field_id}

# Get irrigation priorities
GET /priority-list

# View alerts
GET /alerts
```

---

# Quick Start

```bash
# Clone repository
git clone [https://github.com/ankitaaidev/Jal-Drishti.git]
cd jal-drishti/backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Start API
uvicorn app.main:app --reload
```

Open

```
http://localhost:8000/docs
```

for the interactive API documentation.

---

# Features

|                             |                                                                       |
| --------------------------- | --------------------------------------------------------------------- |
| **Satellite imagery**       | Sentinel-1 SAR and Sentinel-2 Optical imagery via Google Earth Engine |
| **Weather integration**     | Open-Meteo rainfall and temperature data                              |
| **Crop detection**          | Rule-based crop classification using spectral signatures              |
| **Growth stage estimation** | NDVI curve + sowing date heuristic                                    |
| **Moisture stress**         | NDWI and rainfall deficit model                                       |
| **Water deficit**           | FAO-56 crop water balance with Hargreaves ET₀                         |
| **Priority ranking**        | Explainable irrigation priority scoring                               |
| **REST API**                | FastAPI with automatic OpenAPI documentation                          |
| **Dashboard**               | Lightweight HTML + JavaScript frontend                                |

---

# Technology Stack

## Backend

* Python 3.10+
* FastAPI
* SQLAlchemy 2.0
* Pydantic v2
* SQLite / PostgreSQL
* Google Earth Engine API
* Open-Meteo API
* httpx
* NumPy
* scikit-learn
* joblib

## Frontend

* HTML5
* CSS3
* Vanilla JavaScript

---

# API

```
POST /fields
GET  /fields
GET  /fields/{id}

POST /satellite/{id}/refresh
POST /infer/{id}

GET  /priority-list
GET  /alerts
GET  /fields/{id}/timeline
```

---

# System Pipeline

```
Field Polygon
      │
      ▼
Google Earth Engine
(Sentinel-1 + Sentinel-2)
      │
      ▼
Satellite Features
(NDVI • NDWI • VV • VH)
      │
      ▼
Open-Meteo Weather
      │
      ▼
Feature Engineering
      │
      ▼
Crop Classification
      │
      ▼
Growth Stage
      │
      ▼
Moisture Stress
      │
      ▼
Water Deficit (FAO-56)
      │
      ▼
Priority Ranking
      │
      ▼
REST API + Dashboard
```

---

# Project Structure

```text
jal-drishti/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── utils/
│   │   └── main.py
│   ├── scripts/
│   ├── tests/
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   └── dashboard.html
│
├── data/
├── docs/
├── database/
├── models/
└── README.md
```

---

# Requirements

* Python 3.10+
* Google Earth Engine Account
* Internet connection for live satellite imagery
* SQLite (default) or PostgreSQL
* 4 GB+ RAM recommended

---

# Future Work

* Multi-temporal crop classification
* Machine learning crop prediction
* React dashboard
* Interactive maps
* Time-series analytics
* PostGIS support
  
---

# License

This project is licensed under the **MIT License**. See the [LICENSE](https://github.com/ankitaaidev/Jal-Drishti/blob/main/LICENSE) file for details.
