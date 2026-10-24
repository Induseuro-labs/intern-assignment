# Q1 — Image Catalog, Detection, Metadata & Composites

Run: `pip install -r requirements.txt` then `python demo.py`.
It ingests all 8 test images, asserts every scenario below, and writes
`outputs/det_*.png` (detections with boxes) + `outputs/comp1.png` /
`outputs/comp2.png` (composites).

## Architecture

```
detector.py   vision layer: background removal + components + heuristics
catalog.py    entries + manual correction + metadata + composite generation
data.py       deterministic test-image factory (8 brief images, no downloads)
demo.py       runnable check asserting all 7 brief scenarios + visual proof
outputs/      annotated detections and composites
```

## Algorithms

- **Detection**: near-white background removal → foreground connected
  components (one component = one object, so a touching cartoon figure stays
  whole while separated items split) → per-color connected regions per object.
  Regions merge by default and split into per-color objects only when two
  regions overlap substantially yet neither contains the other — this splits
  truly overlapping shapes while keeping a leaf-in-apple-box a detail and body
  parts merged. Shape = bbox fill-ratio + aspect; color = nearest palette
  entry; confidence scales with solidity (0.55 for overlap splits).
- **Metadata**: auto-only fields (`shape`, `label`, snapshot) vs
  user-editable (`name`, `color`, `bbox`, `confidence`) — editing anything
  else raises `ValueError`. Every manual touch sets `corrected=True`;
  corrections persist until `rerun_detection()` is explicitly called.
- **Composites**: `(count, color, noun)` triples + unknown-phrase detection
  ("dragon fruit") resolve against real catalog entries only. Missing assets
  → `ValueError`, never a generated substitute. Assets are bg-stripped and
  pasted on a grid; positions/scales/prompt are stored as provenance and the
  composite is re-ingested through detection as its own entry.

## Examples covered (all asserted in `demo.py`)

1. Successful detection — `apple`→1 red object, `shapes`→4, `colors`→5,
   `apple_orange`→2, `boy`/`girl`→1 each with clothing palette,
   `overlapping_shapes`→3 with overlapping boxes (see `det_*.png`).
2. Failed detection — `blank` yields zero objects, `status="no_objects"`.
3. Manual correction — add the missed apple, rename/recolor/rebox it;
   editing an auto-only field is refused.
4. Persistence — unrelated ingests don't touch corrections; explicit rerun
   discards them.
5. Composite — "one red apple and one orange" reuses the actual catalog
   pixels (pixel-asserted); "dragon fruit" fails loudly.
6. Composite — boy + girl + colored shapes from three assets.
7. Overlapping + empty — asserted throughout, visible in `det_*.png`.

## Trade-offs

Plain-background assumption (true of this catalog; real photos would need a
learned segmenter). Heuristic naming is *expected* to err — manual correction
is the backstop, not an afterthought. Thresholds were calibrated on the 8
test images and sit next to the code. Upgrade path: Grounding-DINO + SAM
behind the same `Catalog` API. No model weights, no API keys, no GPU needed.
