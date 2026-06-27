from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging import setup_logging, get_logger
from app.db.session import engine, Base
from app.api.routes import fields, satellite, inference, irrigation, alerts

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Jal-Drishti API",
    description=(
        "Satellite-driven irrigation decision platform. Crop type, growth "
        "stage, moisture stress, and water deficit are computed from real "
        "Sentinel-1/2 imagery (Google Earth Engine) and real weather data "
        "(Open-Meteo). See README.md for what is and isn't ML in v1."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fields.router)
app.include_router(satellite.router)
app.include_router(inference.router)
app.include_router(irrigation.router)
app.include_router(alerts.router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    logger.info("Jal-Drishti backend started. DB tables ensured.")


@app.get("/")
def root():
    return {
        "service": "jal-drishti-backend",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
