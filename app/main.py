import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import ocr as ocr_router
from app.routes import sheet as sheet_router
from app.routes import worker as worker_router
from app.routes import shift as shift_router
from app.routes import qc as qc_router
from app.services.ocr_service import OCRService
from app.services.sheet_service import SheetService
from app.services.worker_service import WorkerService
from app.services.qc_service import QCService

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Machine Sense Backend Starting ===")
    app.state.ocr_service = OCRService()      # EasyOCR model loads here
    app.state.sheet_service = SheetService()  # YOLOv8 loads per camera on register
    app.state.worker_service = WorkerService()  # YOLOv8 person tracker
    app.state.qc_service = QCService()          # QC form entry storage
    logger.info("=== Startup complete. Ready to accept requests. ===")

    yield  # App runs here

    logger.info("=== Machine Sense Backend Shutting Down ===")
    app.state.ocr_service.shutdown()
    app.state.sheet_service.shutdown()
    app.state.worker_service.shutdown()
    app.state.qc_service.shutdown()


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="Machine Sense API",
    description=(
        "AI Machine & Worker Monitoring System\n\n"
        "- **Phase 1** `/ocr/...` — Read machine counter values using EasyOCR + OpenCV\n"
        "- **Phase 2** `/workers/...` — Worker presence & activity tracking using YOLOv8\n"
        "- **Phase 3** `/sheets/...` — Count sheets using YOLOv8 + ByteTrack"
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — allow Next.js frontend (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------
app.include_router(ocr_router.router)
app.include_router(worker_router.router)
app.include_router(shift_router.router)
app.include_router(sheet_router.router)
app.include_router(qc_router.router)


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "system": "Machine Sense", "phase": "2 (worker tracking)"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
