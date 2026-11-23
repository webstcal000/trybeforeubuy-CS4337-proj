"""HTTP adapter for the local virtual try-on pipeline."""

from __future__ import annotations

import logging
import os
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool

from controller.body_segmentation import BodySegmenter
from controller.image_warping import GarmentWarper
from model.deepfashion import DeepFashionAPI

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = ROOT / "downloads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_DIMENSION = 8_000
CATALOG_LIMIT = 48
CATALOG_PER_CATEGORY = 6
logger = logging.getLogger(__name__)


class ApiProblem(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _image_from_bytes(data: bytes, expected_type: str | None = None) -> np.ndarray:
    signatures = {
        "image/jpeg": data.startswith(b"\xff\xd8\xff"),
        "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
    }
    if expected_type is not None and not signatures.get(expected_type, False):
        raise ApiProblem(
            400,
            "image_type_mismatch",
            "The file contents do not match the selected JPEG or PNG image type.",
        )
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ApiProblem(400, "invalid_image", "That file could not be decoded as an image.")
    height, width = image.shape[:2]
    if (
        width < 64
        or height < 64
        or width > MAX_IMAGE_DIMENSION
        or height > MAX_IMAGE_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise ApiProblem(
            422,
            "invalid_dimensions",
            "Use an image at least 64 × 64 pixels and no larger than 8,000 pixels per side or 40 megapixels.",
        )
    return image


def _image_response(image: np.ndarray, extension: str, media_type: str) -> Response:
    encoded, buffer = cv2.imencode(extension, image)
    if not encoded:
        raise ApiProblem(500, "image_encoding_failed", "The image could not be prepared.")
    return Response(content=buffer.tobytes(), media_type=media_type)


def _build_catalog(dataset_api: DeepFashionAPI) -> dict[int, dict[str, object]]:
    dataset = dataset_api.dataset
    catalog: dict[int, dict[str, object]] = {}
    per_category: dict[str, int] = {}

    for garment_id in dataset.select(partition="train", fully_visible=True):
        clothes_type = int(dataset.clothes_type[garment_id])
        if clothes_type not in (1, 3):
            continue

        item = dataset.get_item(garment_id, load_image=False)
        category = item["category"]
        if per_category.get(category, 0) >= CATALOG_PER_CATEGORY:
            continue

        try:
            image = dataset.read_image(item["image_name"])
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, cv2.error) as error:
            logger.warning("Skipping unreadable DeepFashion item %s: %s", garment_id, error)
            continue
        if image is None:
            logger.warning("Skipping unreadable DeepFashion image for item %s", garment_id)
            continue

        catalog[garment_id] = {
            "id": garment_id,
            "name": category.replace("_", " "),
            "category": category,
        }
        per_category[category] = per_category.get(category, 0) + 1
        if len(catalog) >= CATALOG_LIMIT:
            break
    return catalog


def _find_sample_person(
    dataset_api: DeepFashionAPI,
    catalog: dict[int, dict[str, object]],
) -> np.ndarray | None:
    dataset = dataset_api.dataset
    for garment_id, (partition, clothes_type) in enumerate(
        zip(dataset.partition, dataset.clothes_type)
    ):
        if partition != "train" or garment_id in catalog or int(clothes_type) not in (1, 3):
            continue
        try:
            image = dataset.read_image(dataset.names[garment_id])
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, cv2.error) as error:
            logger.warning("Skipping sample-person candidate %s: %s", garment_id, error)
            continue
        if image is not None and image.shape[0] >= 64 and image.shape[1] >= 64:
            return image
    return None


def _process_try_on(app: FastAPI, image: np.ndarray, garment_id: int) -> np.ndarray:
    with app.state.pipeline_lock:
        if app.state.segmenter is None:
            app.state.segmenter = BodySegmenter()
        segmentation = app.state.segmenter.run(image)
        garment = app.state.dataset_api.fetch_garment(garment_id)
        return GarmentWarper().try_on(image, segmentation, garment)


def create_app(
    dataset_dir: str | Path | None = None,
    sample_person_path: str | Path | None = None,
) -> FastAPI:
    def load_local_assets() -> None:
        root = app.state.dataset_dir
        try:
            dataset_api = DeepFashionAPI(str(root))
            if not (root / "img").is_dir() and not (root / "img.zip").is_file():
                raise FileNotFoundError("DeepFashion image folder or img.zip is missing")
            catalog = _build_catalog(dataset_api)
            if not catalog:
                raise ValueError("No compatible, fully visible training garments were found")
            app.state.dataset_api = dataset_api
            app.state.catalog = catalog
        except (FileNotFoundError, OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
            app.state.dataset_issue = str(error)
            logger.warning("DeepFashion catalog is unavailable: %s", error)

        sample_path = app.state.sample_person_path
        try:
            if not sample_path.is_file():
                raise FileNotFoundError("Sample person image is missing")
            app.state.sample_person_type = "image/png" if sample_path.suffix.lower() == ".png" else "image/jpeg"
            app.state.sample_person_image = _image_from_bytes(
                sample_path.read_bytes(), app.state.sample_person_type
            )
        except (OSError, ApiProblem, cv2.error) as error:
            app.state.sample_issue = str(error)
            logger.info("Sample person image is unavailable: %s", error)
            if not app.state.sample_person_explicit and app.state.dataset_api is not None:
                app.state.sample_person_image = _find_sample_person(
                    app.state.dataset_api, app.state.catalog
                )
                if app.state.sample_person_image is not None:
                    app.state.sample_issue = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        load_local_assets()
        yield

    app = FastAPI(title="TryBeforeUBuy API", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.dataset_dir = Path(
        dataset_dir or os.environ.get("DEEPFASHION_DIR", DOWNLOADS_DIR)
    ).expanduser()
    app.state.sample_person_path = Path(
        sample_person_path
        or os.environ.get("SAMPLE_PERSON_PATH", DOWNLOADS_DIR / "sample-person.jpg")
    ).expanduser()
    app.state.dataset_api = None
    app.state.catalog = {}
    app.state.dataset_issue = None
    app.state.sample_issue = None
    app.state.sample_person_image = None
    app.state.sample_person_type = "image/jpeg"
    app.state.sample_person_explicit = (
        sample_person_path is not None or bool(os.environ.get("SAMPLE_PERSON_PATH"))
    )
    app.state.pipeline_lock = threading.Lock()
    app.state.segmenter = None

    @app.exception_handler(ApiProblem)
    async def api_problem_handler(_: Request, error: ApiProblem) -> JSONResponse:
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, error: StarletteHTTPException) -> JSONResponse:
        code = {
            404: "not_found",
            405: "method_not_allowed",
        }.get(error.status_code, "request_failed")
        message = error.detail if isinstance(error.detail, str) else "The request could not be completed."
        return _error_response(error.status_code, code, message)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            422,
            "invalid_request",
            "Check the request fields and try again.",
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(_: Request, error: Exception) -> JSONResponse:
        logger.error(
            "Unhandled API error",
            exc_info=(type(error), error, error.__traceback__),
        )
        return _error_response(500, "internal_error", "The request could not be completed. Please try again.")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        dataset_ready = app.state.dataset_api is not None
        sample_ready = app.state.sample_issue is None
        issues = []
        if not dataset_ready:
            issues.append({
                "code": "dataset_unavailable",
                "message": "Add the DeepFashion annotations and images to downloads/ or set DEEPFASHION_DIR.",
            })
        if not sample_ready:
            issues.append({
                "code": "sample_unavailable",
                "message": "Add downloads/sample-person.jpg or set SAMPLE_PERSON_PATH. You can still upload a photo.",
            })
        return {
            "service": "ok",
            "dataset_ready": dataset_ready,
            "catalog_ready": bool(app.state.catalog),
            "sample_ready": sample_ready,
            "pipeline_ready": dataset_ready and bool(app.state.catalog),
            "issues": issues,
        }

    @app.get("/api/garments")
    def garments(category: str | None = None) -> dict[str, object]:
        if app.state.dataset_api is None:
            raise ApiProblem(
                503,
                "dataset_unavailable",
                "The local DeepFashion dataset is not ready. Check the downloads/ setup instructions.",
            )
        selected = [
            item for item in app.state.catalog.values()
            if category is None or item["category"].casefold() == category.casefold()
        ]
        categories = sorted({str(item["category"]) for item in app.state.catalog.values()})
        return {
            "garments": [
                {
                    **item,
                    "imageUrl": f"/api/garments/{item['id']}/image",
                }
                for item in selected
            ],
            "categories": categories,
        }

    @app.get("/api/garments/{garment_id}/image")
    def garment_image(garment_id: int) -> Response:
        if garment_id not in app.state.catalog:
            raise ApiProblem(404, "garment_not_found", "That garment is not in the available catalog.")
        dataset = app.state.dataset_api.dataset
        item = dataset.get_item(garment_id, load_image=True)
        if item["image"] is None:
            raise ApiProblem(404, "image_unavailable", "The garment image could not be loaded.")
        response = _image_response(item["image"], ".jpg", "image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response

    @app.get("/api/sample-person")
    def sample_person() -> Response:
        image = app.state.sample_person_image
        if image is None:
            raise ApiProblem(
                503,
                "sample_unavailable",
                "No sample photo is installed. Upload your own image to continue.",
            )
        extension = ".png" if app.state.sample_person_type == "image/png" else ".jpg"
        return _image_response(image, extension, app.state.sample_person_type)

    @app.post("/api/try-on")
    async def try_on(
        person: UploadFile = File(...),
        garment_id: int = Form(...),
    ) -> Response:
        if person.content_type not in ("image/jpeg", "image/png"):
            raise ApiProblem(400, "unsupported_image_type", "Upload a JPEG or PNG photo.")
        data = await person.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ApiProblem(413, "image_too_large", "Choose an image smaller than 10 MB.")
        image = _image_from_bytes(data, person.content_type)
        if app.state.dataset_api is None:
            raise ApiProblem(
                503,
                "dataset_unavailable",
                "The local DeepFashion dataset is not ready. Try again after it has been installed.",
            )
        if garment_id not in app.state.catalog:
            raise ApiProblem(422, "garment_unavailable", "Choose a garment from the available catalog.")

        try:
            result = await run_in_threadpool(_process_try_on, app, image, garment_id)
        except ValueError as error:
            raise ApiProblem(
                422,
                "try_on_not_possible",
                "This photo and garment could not be aligned. Try a clear, front-facing photo.",
            ) from error
        except Exception as error:
            logger.exception("Try-on processing failed")
            raise ApiProblem(
                500,
                "try_on_failed",
                "The try-on could not be completed. Check the photo and try again.",
            ) from error

        return _image_response(result, ".png", "image/png")

    return app


app = create_app()
