"""FastAPI app: REST API + a small browser UI for reviewing and correcting detections.

    python -m imgcat serve            # http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .composite import MissingAssetError
from .service import CatalogService, ProtectedObjectsError

STATIC = Path(__file__).parent / "static"


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ObjectIn(BaseModel):
    name: str
    bbox: BBox
    color: str | None = None
    confidence: float = 1.0
    display_name: str | None = None
    notes: str | None = None


class ObjectPatch(BaseModel):
    name: str | None = None
    display_name: str | None = None
    color: str | None = None
    confidence: float | None = None
    bbox: BBox | None = None
    notes: str | None = None


class ImagePatch(BaseModel):
    display_name: str | None = None
    description: str | None = None
    tags: list[str] | None = None


class ComposeIn(BaseModel):
    instruction: str
    display_name: str | None = None
    verify: bool = False
    use_llm: bool | None = None


def create_app(root: str | Path | None = None, service: CatalogService | None = None) -> FastAPI:
    svc = service or CatalogService(root or os.environ.get("IMGCAT_ROOT", "runtime/catalog"))
    app = FastAPI(title="Image catalog")

    def image_or_404(image_id: str):
        try:
            return svc.store.get_image(image_id)
        except KeyError:
            raise HTTPException(404, f"no image {image_id}")

    def object_or_404(object_id: str):
        try:
            return svc.store.get_object(object_id)
        except KeyError:
            raise HTTPException(404, f"no object {object_id}")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/images")
    def list_images():
        return [i.to_dict() for i in svc.store.list_images()]

    @app.post("/api/images", status_code=201)
    async def upload(file: UploadFile = File(...), display_name: str | None = Form(None),
                     detect: bool = Form(True)):
        suffix = Path(file.filename or "upload.png").suffix or ".png"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{Path(file.filename or 'upload').stem}{suffix}"
            path.write_bytes(await file.read())
            try:
                img = svc.add_image(path, display_name or path.stem, detect=detect)
            except Exception as exc:  # unreadable file
                raise HTTPException(400, f"cannot read image: {exc}")
        return img.to_dict()

    @app.get("/api/images/{image_id}")
    def get_image(image_id: str):
        return image_or_404(image_id).to_dict(include_deleted=True)

    @app.get("/api/images/{image_id}/file")
    def image_file(image_id: str):
        image_or_404(image_id)
        return FileResponse(svc.store.image_path(image_id))

    @app.patch("/api/images/{image_id}")
    def patch_image(image_id: str, body: ImagePatch):
        image_or_404(image_id)
        svc.store.update_image_user_fields(image_id, **body.model_dump(exclude_none=True))
        return svc.store.get_image(image_id).to_dict()

    @app.delete("/api/images/{image_id}", status_code=204)
    def delete_image(image_id: str):
        image_or_404(image_id)
        svc.store.delete_image(image_id)

    @app.post("/api/images/{image_id}/detect")
    def detect(image_id: str, force: bool = False):
        image_or_404(image_id)
        try:
            return svc.run_detection(image_id, force=force).to_dict()
        except ProtectedObjectsError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/images/{image_id}/runs")
    def runs(image_id: str):
        image_or_404(image_id)
        return svc.store.runs(image_id)

    @app.post("/api/images/{image_id}/objects", status_code=201)
    def add_object(image_id: str, body: ObjectIn):
        image_or_404(image_id)
        data = body.model_dump(exclude_none=True)
        try:
            return svc.add_object(image_id, data.pop("name"), data.pop("bbox"), **data).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.patch("/api/objects/{object_id}")
    def patch_object(object_id: str, body: ObjectPatch):
        object_or_404(object_id)
        try:
            return svc.update_object(object_id, **body.model_dump(exclude_none=True)).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.delete("/api/objects/{object_id}", status_code=204)
    def delete_object(object_id: str):
        object_or_404(object_id)
        svc.delete_object(object_id)

    @app.post("/api/objects/{object_id}/restore")
    def restore_object(object_id: str):
        object_or_404(object_id)
        return svc.restore_object(object_id).to_dict()

    @app.post("/api/objects/{object_id}/revert/{field}")
    def revert(object_id: str, field: str):
        object_or_404(object_id)
        return svc.revert_field(object_id, field).to_dict()

    @app.post("/api/composites", status_code=201)
    def compose(body: ComposeIn):
        try:
            return svc.compose(body.instruction, body.display_name, use_llm=body.use_llm,
                               verify=body.verify).to_dict()
        except MissingAssetError as exc:
            raise HTTPException(422, {"error": "missing_assets", "missing": exc.missing, "message": str(exc)})

    return app
