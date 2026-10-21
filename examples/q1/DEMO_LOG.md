# Question 1 demo log

## Scenario 1 - detect objects (names, colours, confidence, bounding boxes)

```
apple: A single red apple on a white background
  status=detected backend=classical-cv-v1
  - apple      color=red     conf=0.78 bbox=(90,80,221,251) origin=detector
orange: A single orange fruit on a white background
  status=detected backend=classical-cv-v1
  - orange     color=orange  conf=0.85 bbox=(90,100,221,221) origin=detector
boy: A cartoon boy standing
  status=detected backend=classical-cv-v1
  - person     color=blue    conf=0.60 bbox=(120,46,161,333) origin=detector
girl: A cartoon girl standing
  status=detected backend=classical-cv-v1
  - person     color=pink    conf=0.60 bbox=(110,40,181,346) origin=detector
shapes: Circle, square, triangle, and rectangle
  status=detected backend=classical-cv-v1
  - circle     color=red     conf=1.00 bbox=(40,80,141,141) origin=detector
  - square     color=blue    conf=1.00 bbox=(240,80,141,141) origin=detector
  - triangle   color=green   conf=1.00 bbox=(431,71,159,150) origin=detector
  - rectangle  color=purple  conf=1.00 bbox=(650,105,171,91) origin=detector
colors: Red, blue, green, yellow, and purple color swatches
  status=detected backend=classical-cv-v1
  - square     color=red     conf=1.00 bbox=(30,50,101,101) origin=detector
  - square     color=blue    conf=1.00 bbox=(165,50,101,101) origin=detector
  - square     color=green   conf=1.00 bbox=(300,50,101,101) origin=detector
  - square     color=yellow  conf=1.00 bbox=(435,50,101,101) origin=detector
  - square     color=purple  conf=1.00 bbox=(570,50,101,101) origin=detector
apple_orange: An apple beside an orange
  status=detected backend=classical-cv-v1
  - apple      color=red     conf=0.78 bbox=(90,80,221,251) origin=detector
  - orange     color=orange  conf=0.85 bbox=(470,100,221,221) origin=detector
overlapping_shapes: Overlapping colored geometric shapes
  status=detected backend=classical-cv-v1
  - circle     color=red     conf=0.80 bbox=(60,80,192,201) origin=detector
  - square     color=blue    conf=0.89 bbox=(200,140,161,161) origin=detector
  - triangle   color=green   conf=1.00 bbox=(300,92,180,169) origin=detector
blank: Empty white image - nothing to detect
  status=no_objects backend=classical-cv-v1
faint: A near-white circle on white - expected detector miss
  status=no_objects backend=classical-cv-v1
cluttered: Apple on a noisy, photo-like background - expected detector failure
  status=failed backend=classical-cv-v1 error=non-uniform background (border spread 67 > 25); classical segmentation unreliable - add objects manually or use the vision backend
```

## Scenario 2 - manually add an object the detector missed (`faint`)

```
added circle origin=user bbox={'x': 100, 'y': 100, 'w': 201, 'h': 201}
added apple to 'cluttered' (detector failed on noisy background)
  status=no_objects backend=classical-cv-v1
  - circle     color=white   conf=1.00 bbox=(100,100,201,201) origin=user
  status=failed backend=classical-cv-v1 error=non-uniform background (border spread 67 > 25); classical segmentation unreliable - add objects manually or use the vision backend
  - apple      color=red     conf=1.00 bbox=(110,100,181,211) origin=user
```

## Scenario 3 - edit name / colour / confidence / bbox

```
boy: 'person' -> 'boy', auto_original kept: {'label': 'person', 'color': 'blue', 'confidence': 0.6, 'x': 120, 'y': 46, 'w': 161, 'h': 333}
girl: 'person' -> 'girl', auto_original kept: {'label': 'person', 'color': 'pink', 'confidence': 0.6, 'x': 110, 'y': 40, 'w': 181, 'h': 346}
apple: bbox {'x': 90, 'y': 80, 'w': 221, 'h': 251} -> {'x': 85, 'y': 75, 'w': 231, 'h': 261}, color red -> dark red
```

## Scenario 4 - corrections persist until detection is explicitly rerun

```
run_detection('boy') refused: 'boy' has manual corrections; pass force=True (optionally keep_manual=True) to explicitly rerun detection
redetect_all(): {"apple": "skipped (manual corrections)", "orange": "detected", "boy": "skipped (manual corrections)", "girl": "skipped (manual corrections)", "shapes": "detected", "colors": "detected", "apple_orange": "detected", "overlapping_shapes": "detected", "blank": "no_objects", "faint": "skipped (manual corrections)", "cluttered": "skipped (manual corrections)"}
boy label after batch reprocess: boy
apple rerun with keep_manual=True -> objects: [('apple', 'detector', True), ('apple', 'detector', False)]
```

## Scenario 5 - composite from the existing apple and orange assets

```
{
  "request": "Create an image containing one red apple and one orange",
  "parser": "rules",
  "items": [
    {
      "term": "apple",
      "count": 1,
      "color": "red"
    },
    {
      "term": "orange",
      "count": 1,
      "color": null
    }
  ],
  "layout": "row, bottom-aligned, size prior per category",
  "generated_pixels": false,
  "sources": [
    {
      "image_id": "e9f84262482b",
      "image_name": "apple",
      "object_id": "d3ac383e23aa",
      "label": "apple",
      "source_bbox": {
        "x": 90,
        "y": 80,
        "w": 221,
        "h": 251
      },
      "scale": 0.502,
      "placed_bbox": {
        "x": 40,
        "y": 40,
        "w": 111,
        "h": 126
      },
      "z": 0
    },
    {
      "image_id": "b492be21009e",
      "image_name": "orange",
      "object_id": "b3fc6bb34332",
      "label": "orange",
      "source_bbox": {
        "x": 90,
        "y": 100,
        "w": 221,
        "h": 221
      },
      "scale": 0.57,
      "placed_bbox": {
        "x": 191,
        "y": 40,
        "w": 126,
        "h": 126
      },
      "z": 1
    }
  ]
}
```

## Scenario 6 - composite with a boy, a girl and coloured shapes

```
  boy        from boy      scale=0.901 placed={'x': 40, 'y': 40, 'w': 145, 'h': 300} z=0
  girl       from girl     scale=0.867 placed={'x': 225, 'y': 40, 'w': 157, 'h': 300} z=1
  triangle   from shapes   scale=0.9 placed={'x': 756, 'y': 205, 'w': 143, 'h': 135} z=2
  circle     from shapes   scale=0.9 placed={'x': 422, 'y': 213, 'w': 127, 'h': 127} z=3
  square     from shapes   scale=0.9 placed={'x': 589, 'y': 213, 'w': 127, 'h': 127} z=4
  rectangle  from shapes   scale=0.9 placed={'x': 939, 'y': 258, 'w': 154, 'h': 82} z=5
```

## Scenario 7 - overlapping objects, empty image, and a missing asset

```
  status=detected backend=classical-cv-v1
  - circle     color=red     conf=0.80 bbox=(60,80,192,201) origin=detector
  - square     color=blue    conf=0.89 bbox=(200,140,161,161) origin=detector
  - triangle   color=green   conf=1.00 bbox=(300,92,180,169) origin=detector
  status=no_objects backend=classical-cv-v1
compose('an apple and a banana') refused: no catalog asset for: banana. Add or label one first (available: apple, apple_orange, boy, circle, cluttered, colors, faint, girl, orange, overlapping_shapes, rectangle, shapes, square, triangle)
```

