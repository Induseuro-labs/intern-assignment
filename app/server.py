"""
FastAPI Backend Server & REST API for Induseuro-Labs Engineering Console.
Serves the custom Skeuomorphic / Claymorphic tactile web frontend on port 8000,
and exposes full REST APIs for Question 1 and Question 2.
"""

from __future__ import annotations
import os
import sys
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from question_1_image_catalog.catalog.models import (
    CatalogEntry, DetectedObject, BoundingBox, ProtectedObjectsError
)
from question_1_image_catalog.catalog.storage import ImageCatalog
from question_1_image_catalog.detector.detector import HybridObjectDetector
from question_1_image_catalog.compositor.engine import CompositeEngine

from question_2_duplicate_prevention.corpus.models import (
    Question, QuestionGenerationRequest, ClassificationCategory
)
from question_2_duplicate_prevention.corpus.repository import QuestionRepository
from question_2_duplicate_prevention.detector.detector import IndependentDuplicateDetector
from question_2_duplicate_prevention.generator.generator import QuestionGeneratorWithAvoidance
from question_2_duplicate_prevention.evaluator.evaluator import QuestionCorpusEvaluator

app = FastAPI(title="Induseuro-Labs Engineering Console API", version="2.0.0")

# Directories
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
Q1_DATA_DIR = os.path.join(PROJECT_ROOT, "question_1_image_catalog", "data")
Q1_COMPOSITES_DIR = os.path.join(PROJECT_ROOT, "question_1_image_catalog", "catalog", "composites")
Q1_STORAGE_PATH = os.path.join(PROJECT_ROOT, "question_1_image_catalog", "catalog", "app_catalog.json")
Q2_STORAGE_PATH = os.path.join(PROJECT_ROOT, "question_2_duplicate_prevention", "corpus", "app_questions.json")
CALIBRATION_PATH = os.path.join(PROJECT_ROOT, "question_2_duplicate_prevention", "data", "calibration_pairs.json")

# Singletons
catalog = ImageCatalog(storage_path=Q1_STORAGE_PATH)
detector_q1 = HybridObjectDetector(use_yolo=False)
compositor = CompositeEngine(catalog=catalog, output_dir=Q1_COMPOSITES_DIR)

repo_q2 = QuestionRepository(storage_path=Q2_STORAGE_PATH)
detector_q2 = IndependentDuplicateDetector()
generator_q2 = QuestionGeneratorWithAvoidance(detector=detector_q2)
evaluator_q2 = QuestionCorpusEvaluator(detector=detector_q2)

# Auto-seed if empty
if len(catalog.list_entries()) == 0:
    from question_1_image_catalog.data.create_test_assets import generate_all
    generate_all()
    seed_files = [
        ("apple", "apple.png"), ("orange", "orange.png"),
        ("boy", "boy.png"), ("girl", "girl.png"),
        ("shapes", "shapes.png"), ("colors", "colors.png"),
        ("apple_orange", "apple_orange.png"),
        ("overlapping_shapes", "overlapping_shapes.png"),
        ("empty_canvas", "empty_canvas.png")
    ]
    from PIL import Image
    for slug, fname in seed_files:
        fpath = os.path.join(Q1_DATA_DIR, fname)
        if os.path.exists(fpath):
            objs = detector_q1.detect(fpath)
            im = Image.open(fpath)
            catalog.add_entry(CatalogEntry(
                id=slug,
                image_name=fname,
                file_path=fpath,
                width=im.width,
                height=im.height,
                format=im.format or "PNG",
                file_size_bytes=os.path.getsize(fpath),
                detection_status="completed" if objs else "no_objects_found",
                objects_source="auto",
                objects=objs
            ))

if len(repo_q2.list_questions()) == 0:
    for q in QuestionRepository.get_seed_test_corpus():
        repo_q2.add_question(q)
    repo_q2.save()


# -----------------------------------------------------------------------------
# Static Image Serving Route
# -----------------------------------------------------------------------------
@app.get("/images/{category}/{filename}")
async def serve_image(category: str, filename: str):
    candidates = []
    if category == "data":
        candidates.append(os.path.join(Q1_DATA_DIR, filename))
    elif category == "composites":
        candidates.append(os.path.join(Q1_COMPOSITES_DIR, filename))
    elif category == "uploaded":
        candidates.append(os.path.join(Q1_DATA_DIR, "uploaded", filename))

    candidates.extend([
        os.path.join(Q1_DATA_DIR, filename),
        os.path.join(Q1_DATA_DIR, "uploaded", filename),
        os.path.join(Q1_COMPOSITES_DIR, filename),
    ])
    for target in candidates:
        if os.path.exists(target):
            return FileResponse(target)
    raise HTTPException(status_code=404, detail="Image file not found")


# -----------------------------------------------------------------------------
# Question 1: REST Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/q1/entries")
def get_catalog_entries():
    entries = catalog.list_entries()
    return [e.model_dump() for e in entries]

@app.get("/api/q1/entry/{entry_id}")
def get_catalog_entry(entry_id: str):
    e = catalog.get_entry(entry_id)
    if not e:
        raise HTTPException(status_code=404, detail=f"Entry '{entry_id}' not found")
    return e.model_dump()

@app.post("/api/q1/add")
async def add_image(
    name: str = Form(...),
    run_detection: bool = Form(True),
    file: UploadFile = File(...)
):
    upload_dir = os.path.join(Q1_DATA_DIR, "uploaded")
    os.makedirs(upload_dir, exist_ok=True)
    out_path = os.path.join(upload_dir, f"{name}.png")
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    from PIL import Image
    im = Image.open(out_path)
    detected = []
    status = "pending"
    if run_detection:
        detected = detector_q1.detect(out_path)
        status = "completed" if detected else "no_objects_found"

    entry = CatalogEntry(
        id=name,
        image_name=f"{name}.png",
        file_path=out_path,
        width=im.width,
        height=im.height,
        format="PNG",
        file_size_bytes=os.path.getsize(out_path),
        detection_status=status,
        objects_source="auto",
        objects=detected
    )
    catalog.add_entry(entry)
    return entry.model_dump()

class EditObjectRequest(BaseModel):
    entry_id: str
    object_id: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[List[int]] = None # [x, y, w, h]
    colors: Optional[List[str]] = None
    notes: Optional[str] = None

@app.post("/api/q1/edit-object")
def edit_object(req: EditObjectRequest):
    bbox_tuple = tuple(req.bbox) if req.bbox and len(req.bbox) == 4 else None
    obj = catalog.edit_object(
        entry_id=req.entry_id,
        object_id=req.object_id,
        label=req.label,
        confidence=req.confidence,
        bbox=bbox_tuple,
        colors=req.colors,
        notes=req.notes
    )
    if not obj:
        raise HTTPException(status_code=404, detail="Entry or object not found")
    return obj.model_dump()

class AddManualObjectRequest(BaseModel):
    entry_id: str
    label: str
    bbox: List[int] # [x, y, w, h]
    colors: Optional[List[str]] = None
    confidence: float = 1.0
    notes: Optional[str] = None

@app.post("/api/q1/add-object")
def add_manual_object(req: AddManualObjectRequest):
    bbox_tuple = tuple(req.bbox) if req.bbox and len(req.bbox) == 4 else (0, 0, 100, 100)
    obj = catalog.add_manual_object(
        entry_id=req.entry_id,
        label=req.label,
        bbox=bbox_tuple,
        colors=req.colors or [],
        confidence=req.confidence,
        notes=req.notes
    )
    if not obj:
        raise HTTPException(status_code=404, detail="Entry not found")
    return obj.model_dump()

class DeleteObjectRequest(BaseModel):
    entry_id: str
    object_id: str

@app.post("/api/q1/delete-object")
def delete_object(req: DeleteObjectRequest):
    ok = catalog.remove_object(req.entry_id, req.object_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Object or entry not found")
    return {"status": "deleted", "object_id": req.object_id}

class RedetectRequest(BaseModel):
    entry_id: str
    force: bool = False

@app.post("/api/q1/redetect")
def redetect(req: RedetectRequest):
    entry = catalog.get_entry(req.entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entry '{req.entry_id}' not found")

    target_path = entry.file_path
    if not os.path.exists(target_path):
        fname = entry.image_name or os.path.basename(target_path.replace("\\", "/"))
        for cand_dir in [Q1_DATA_DIR, os.path.join(Q1_DATA_DIR, "uploaded"), Q1_COMPOSITES_DIR]:
            cand = os.path.join(cand_dir, fname)
            if os.path.exists(cand):
                target_path = cand
                entry.file_path = cand
                catalog.save()
                break

    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Image file for entry '{req.entry_id}' not found on server at {target_path}")

    try:
        fresh = detector_q1.detect(target_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed on '{target_path}': {str(e)}")

    try:
        updated = catalog.update_detection(entry.id, fresh, force=req.force)
        return updated.model_dump()
    except ProtectedObjectsError as err:
        raise HTTPException(status_code=409, detail=str(err))

class ComposeRequest(BaseModel):
    prompt: str
    canvas_w: int = 1000
    canvas_h: int = 600
    name: Optional[str] = None
    verify: bool = True

@app.post("/api/q1/compose")
def compose(req: ComposeRequest):
    try:
        res = compositor.generate_composite(
            prompt=req.prompt,
            canvas_size=(req.canvas_w, req.canvas_h),
            composite_name=req.name or None,
            verify=req.verify
        )
        return res.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# Question 2: REST Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/q2/questions")
def get_questions(concept: Optional[str] = None):
    qs = repo_q2.list_questions(concept=concept)
    return [q.model_dump() for q in qs]

class AddQuestionRequest(BaseModel):
    text: str
    concept: str = "optics"
    tags: Dict[str, Any] = {}

@app.post("/api/q2/add-question")
def add_question(req: AddQuestionRequest):
    new_id = f"Q_USR_{len(repo_q2.list_questions())+1:03d}"
    q = Question(
        id=new_id,
        text=req.text.strip(),
        concept=req.concept.strip(),
        tags=req.tags,
        source="manual_entry"
    )
    repo_q2.add_question(q)
    repo_q2.save()
    return q.model_dump()

class CheckDuplicateRequest(BaseModel):
    candidate: str
    baseline: Optional[str] = None

@app.post("/api/q2/check")
def check_duplicate(req: CheckDuplicateRequest):
    if req.baseline and req.baseline.strip():
        base = [Question(id="BASE", text=req.baseline.strip(), concept="General")]
        res = detector_q2.check_candidate(req.candidate.strip(), base)
    else:
        corpus = repo_q2.list_questions()
        res = detector_q2.check_candidate(req.candidate.strip(), corpus)
    return res.model_dump()

class GenerateQuestionsRequest(BaseModel):
    concept: str = "reflection of light"
    subject: str = "Physics"
    num_questions: int = 3
    context_exemplar_limit: int = 4

@app.post("/api/q2/generate")
def generate_questions_api(req: GenerateQuestionsRequest):
    r = QuestionGenerationRequest(
        subject=req.subject,
        concept=req.concept,
        num_questions=req.num_questions,
        avoid_duplicates=True,
        context_exemplar_limit=req.context_exemplar_limit
    )
    corpus = repo_q2.list_questions()
    res = generator_q2.generate_questions(r, corpus)
    return {
        "prompt": res["prompt"],
        "exemplars_provided": res["exemplars_provided"],
        "exhausted_facets": res["exhausted_facets"],
        "accepted_questions": [q.model_dump() for q in res["accepted_questions"]],
        "rejected_duplicates": [d.model_dump() for d in res["rejected_duplicates"]],
        "safeguard_checks": [c.model_dump() for c in res["safeguard_checks"]]
    }

@app.get("/api/q2/eval-corpus")
def eval_corpus():
    res = evaluator_q2.evaluate_official_corpus()
    return res

@app.get("/api/q2/calibration")
def eval_calibration():
    res = evaluator_q2.evaluate_calibration_pairs(CALIBRATION_PATH)
    return res

@app.get("/api/q2/benchmark-scale")
def benchmark_scale():
    res = evaluator_q2.benchmark_vector_retrieval_scale()
    return res


# -----------------------------------------------------------------------------
# Static Frontend Serving
# -----------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Induseuro-Labs Engineering Console API is live. Static frontend compiling."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
