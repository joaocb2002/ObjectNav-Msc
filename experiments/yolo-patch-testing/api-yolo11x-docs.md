# Results and Boxes Classes Documentation

## Results Class

A class for storing and manipulating inference results.

This class provides methods for accessing, manipulating, and visualizing inference results from various Ultralytics models, including detection, segmentation, classification, and pose estimation.

### Attributes

- **`orig_img (numpy.ndarray)`**: The original image as a numpy array.
- **`orig_shape (Tuple[int, int])`**: Original image shape in (height, width) format.
- **`boxes (Boxes | None)`**: Detected bounding boxes.
- **`masks (Masks | None)`**: Segmentation masks.
- **`probs (Probs | None)`**: Classification probabilities.
- **`keypoints (Keypoints | None)`**: Detected keypoints.
- **`obb (OBB | None)`**: Oriented bounding boxes.
- **`speed (dict)`**: Dictionary containing inference speed information.
- **`names (dict)`**: Dictionary mapping class indices to class names.
- **`path (str)`**: Path to the input image file.
- **`save_dir (str | None)`**: Directory to save results.

### Methods

- **`update()`**: Updates the Results object with new detection data.
- **`cpu()`**: Returns a copy of the Results object with all tensors moved to CPU memory.
- **`numpy()`**: Converts all tensors in the Results object to numpy arrays.
- **`cuda()`**: Moves all tensors in the Results object to GPU memory.
- **`to(device, dtype)`**: Moves all tensors to the specified device and dtype.
- **`new()`**: Creates a new Results object with the same image, path, names, and speed attributes.
- **`plot()`**: Plots detection results on an input RGB image.
- **`show()`**: Displays the image with annotated inference results.
- **`save()`**: Saves annotated inference results image to file.
- **`verbose()`**: Returns a log string for each task in the results.
- **`save_txt()`**: Saves detection results to a text file.
- **`save_crop()`**: Saves cropped detection images to specified directory.
- **`summary()`**: Converts inference results to a summarized dictionary.
- **`to_df()`**: Converts detection results to a Pandas Dataframe.
- **`to_json()`**: Converts detection results to JSON format.
- **`to_csv()`**: Converts detection results to a CSV format.
- **`to_xml()`**: Converts detection results to XML format.
- **`to_html()`**: Converts detection results to HTML format.
- **`to_sql()`**: Converts detection results to an SQL-compatible format.

### Example Usage

```python
results = model("path/to/image.jpg")
result = results[0]  # Get the first result
boxes = result.boxes  # Get the boxes for the first result
masks = result.masks  # Get the masks for the first result

for result in results:
    result.plot()  # Plot detection result
```

# Boxes Class

A class for managing and manipulating detection boxes.

This class provides functionality for handling detection boxes, including their coordinates, confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers methods for easy manipulation and conversion between different coordinate systems.

## Attributes

- **`data (torch.Tensor | numpy.ndarray)`**: The raw tensor containing detection boxes and associated data.
- **`orig_shape (Tuple[int, int])`**: The original image dimensions (height, width).
- **`is_track (bool)`**: Indicates whether tracking IDs are included in the box data.
- **`xyxy (torch.Tensor | numpy.ndarray)`**: Boxes in `[x1, y1, x2, y2]` format.
- **`conf (torch.Tensor | numpy.ndarray)`**: Confidence scores for each box.
- **`cls (torch.Tensor | numpy.ndarray)`**: Class labels for each box.
- **`id (torch.Tensor | None)`**: Tracking IDs for each box (if available).
- **`xywh (torch.Tensor | numpy.ndarray)`**: Boxes in `[x, y, width, height]` format.
- **`xyxyn (torch.Tensor | numpy.ndarray)`**: Normalized `[x1, y1, x2, y2]` boxes relative to `orig_shape`.
- **`xywhn (torch.Tensor | numpy.ndarray)`**: Normalized `[x, y, width, height]` boxes relative to `orig_shape`.

## Methods

- **`cpu()`**: Returns a copy of the object with all tensors on CPU memory.
- **`numpy()`**: Returns a copy of the object with all tensors as numpy arrays.
- **`cuda()`**: Returns a copy of the object with all tensors on GPU memory.
- **`to(*args, **kwargs)`**: Returns a copy of the object with tensors on specified device and dtype.

## Example Usage

```python
import torch

boxes_data = torch.tensor([
    [100, 50, 150, 100, 0.9, 0],
    [200, 150, 300, 250, 0.8, 1]
])
orig_shape = (480, 640)  # height, width

boxes = Boxes(boxes_data, orig_shape)

print(boxes.xyxy)
print(boxes.conf)
print(boxes.cls)
print(boxes.xywhn)

