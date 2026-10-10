from fastapi.testclient import TestClient

from conftest import IMAGES
from imgcat.api import create_app


def test_api_end_to_end(svc):
    client = TestClient(create_app(service=svc))
    with open(IMAGES / "shapes.png", "rb") as f:
        r = client.post("/api/images", files={"file": ("shapes.png", f, "image/png")}, data={"display_name": "shapes"})
    assert r.status_code == 201
    img = r.json()
    assert img["auto"]["detection_status"] == "detected" and len(img["objects"]) == 4

    obj = img["objects"][0]
    r = client.patch(f"/api/objects/{obj['id']}", json={"name": "ball", "confidence": 0.9})
    assert r.status_code == 200 and r.json()["effective"]["name"] == "ball"
    assert client.patch(f"/api/objects/{obj['id']}", json={"confidence": 3}).status_code == 422

    r = client.post(f"/api/images/{img['id']}/objects",
                    json={"name": "star", "bbox": {"x": 10, "y": 10, "width": 30, "height": 30}})
    assert r.status_code == 201 and r.json()["origin"] == "user"

    assert client.post(f"/api/images/{img['id']}/detect").status_code == 409
    r = client.post(f"/api/images/{img['id']}/detect?force=true")
    assert r.status_code == 200 and r.json()["auto"]["objects_source"] == "auto"

    r = client.post("/api/composites", json={"instruction": "a blue square and a banana", "use_llm": False})
    assert r.status_code == 422 and r.json()["detail"]["missing"] == ["banana"]
    r = client.post("/api/composites", json={"instruction": "a blue square and a red circle", "use_llm": False})
    assert r.status_code == 201 and r.json()["auto"]["kind"] == "composite"
    assert client.get(f"/api/images/{r.json()['id']}/file").status_code == 200
    assert client.get("/").status_code == 200
