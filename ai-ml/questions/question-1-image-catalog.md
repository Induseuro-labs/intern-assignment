# Question 1: Image Catalog, Object Detection, Metadata, and Composite Image Generation

## 1. Image Catalog

Design a system that maintains a reusable collection of images. Each catalog entry represents a stored image and its associated metadata.

## 2. Automatic Object Detection

When an image is added, the system should automatically detect objects or regions within it and extract relevant information.

Object detection must work for both cartoonized (illustrated or stylized) images and non-cartoonized (photographic or realistic) images.

If detection fails or produces incorrect results, users must be able to manually add, edit, or remove detected objects and save their corrections.

Manual corrections must remain the source of truth unless a user explicitly requests detection to run again.

## 3. Metadata Management

Store metadata associated with each image and its detected objects. Examples include object names, confidence scores, bounding boxes (x, y, width, height), image dimensions, display names, and detection status.

Clearly distinguish automatically generated fields from user-editable fields.

## 4. Composite Image Generation

Allow users to generate a new image from existing catalog images using a natural-language description.

For example, given separate catalog images of a red apple and an orange, the instruction "Create an image containing one red apple and one orange" should produce a composite using those existing image assets.

The system should support appropriate positioning, scaling, layering, or other composition techniques. Required catalog assets must not be silently replaced with newly generated approximations.

Save each generated composite as a new catalog entry with its own metadata.

## 5. Implementation Requirements

Provide working code and explain the architecture, algorithms, technology choices, and trade-offs. Include examples covering:

- Successful detection
- Failed detection
- Manual correction
- Composite generation

You may use local libraries, GPUs, hosted AI services, or a combination of these. Do not assume that processing must be CPU-only or entirely local.

## 6. Test Data

Use the images and scenarios below to validate object detection, manual correction, metadata, and composite generation.

### Image catalog

| Image name | Description | Expected objects / metadata |
|------------|-------------|---------------------------|
| `apple` | A single red apple on a white background | Object: apple; color: red; bounding box |
| `orange` | A single orange fruit on a white background | Object: orange; color: orange; bounding box |
| `boy` | A cartoon boy standing | Object: boy; clothing colors; bounding box |
| `girl` | A cartoon girl standing | Object: girl; clothing colors; bounding box |
| `shapes` | Circle, square, triangle, and rectangle | Four objects; shape labels; bounding boxes |
| `colors` | Red, blue, green, yellow, and purple color swatches | Five regions; color labels; bounding boxes |
| `apple_orange` | An apple beside an orange | Two objects; individual bounding boxes |
| `overlapping_shapes` | Overlapping colored geometric shapes | Shape labels; colors; overlapping bounding boxes |

### Test scenarios

1. Detect objects and extract their names, colors, confidence scores, and bounding boxes.
2. Manually add an object when detection misses it.
3. Edit an object's name, color, confidence, or bounding box.
4. Verify manual corrections persist until detection is explicitly rerun.
5. Generate a composite using the existing `apple` and `orange` assets.
6. Generate a composite containing a boy, a girl, and colored shapes.
7. Test overlapping objects and images containing no detectable objects.
