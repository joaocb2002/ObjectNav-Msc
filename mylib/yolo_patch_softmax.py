# Patch YOLO's inference pipeline to retain full class probability vectors (post-sigmoid),
# normalize them, and preserve them through non-max suppression and final result formatting.

import ultralytics.engine.results
import ultralytics.utils.ops
import torch
import torchvision
from ultralytics.utils.ops import xywh2xyxy, LOGGER, nms_rotated
import ultralytics.engine.results as results_mod
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils.ops import scale_boxes, convert_torch2numpy_batch

# Softmax temperature for class probabilities
TEMPERATURE = 2.4

# Load original Boxes class
OriginalBoxes = results_mod.Boxes

# Override Boxes class to interpret .conf and .cls from full class vector
class PatchedBoxes(OriginalBoxes):
    def __init__(self, boxes, orig_shape):
        # Same original init       
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]

        self.orig_shape = orig_shape
        self.is_track = False
        self.num_classes = 0

        # Prepare for new output format
        if n == 6:
            self.format = 'xyxy_conf_cls'
        elif n == 7:
            self.format = 'xyxy_conf_cls_track'
            self.is_track = True
        else:
            self.format = 'xyxy_conf_cls_probs'
            self.num_classes = n - 6

        self.data = boxes

    @property
    def conf(self):
        if self.data.shape[1] > 6:
            return self.data[:, 6:].max(1, keepdim=True).values
        return self.data[:, 4:5]

    @property
    def cls(self):
        if self.data.shape[1] > 6:
            return self.data[:, 6:].argmax(1).to(torch.int)
        return self.data[:, 5].to(torch.int)
    
    @property
    def probs(self):       
        return self.data[:, 6:] if self.data.shape[1] > 6 else None 

# Replace Results.Boxes with the patched version
results_mod.Boxes = PatchedBoxes
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #

# Custom non-max suppression function to preserve full class confidence vectors
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
):
    assert 0 <= conf_thres <= 1
    assert 0 <= iou_thres <= 1

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4     # number of masks or extra features (if any)
    mi = 4 + nc                           # used to slice prediction before masks
    xc = prediction[:, 4:mi].amax(1).sigmoid() > conf_thres # confidence mask
    prediction = prediction.transpose(-1, -2)

    # Convert boxes from xywh to xyxy format
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)

    # Initialize output list (per image in batch)
    output = [torch.zeros((0, 6 + nc + nm), device=prediction.device)] * bs

    # Iterate over each image in the batch
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # Filter by confidence mask

        # Add any provided ground truth labels
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box coords
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # one-hot class
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # Split into box coordinates, class confidences, and masks
        box, cls_conf, mask = x.split((4, nc, nm), 1)

        # Extract top-1 class and its confidence (for NMS ranking)
        conf, j = cls_conf.max(1, keepdim=True)
        conf = conf.sigmoid()  # apply sigmoid to class scores

        # Apply softmax to class confidence scores
        cls_conf = torch.softmax(cls_conf / TEMPERATURE, dim=1)  # apply softmax to class scores

        # Retain full class confidence vector after max
        x = torch.cat((box, conf, j.float(), cls_conf, mask), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        n = x.shape[0]
        if not n:
            continue 

        # Keep only top-n scoring boxes to speed up NMS
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Compute offsets for class-aware NMS (if agnostic=False)
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]

        # Perform NMS based on rotated or standard boxes
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c
            i = torchvision.ops.nms(boxes, scores, iou_thres)

        output[xi] = x[i[:max_det]]

        # if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 1e9:
        #     LOGGER.warning("High memory usage detected during NMS (>1GB allocated)")

    return output


# Apply custom NMS
ultralytics.utils.ops.non_max_suppression = non_max_suppression
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #

# Patch postprocess to use custom NMS and scaled boxes with full class confidence retention
def patched_postprocess(self, preds, img, orig_imgs, **kwargs):
    preds = ultralytics.utils.ops.non_max_suppression(
        preds,
        self.args.conf,
        self.args.iou,
        self.args.classes,
        self.args.agnostic_nms,
        max_det=self.args.max_det,
        nc=len(self.model.names),
        end2end=getattr(self.model, "end2end", False),
        rotated=self.args.task == "obb",
    )

    if not isinstance(orig_imgs, list):
        orig_imgs = convert_torch2numpy_batch(orig_imgs)

    results = []
    for i, pred in enumerate(preds):
        if len(pred) > 0:
            pred[:, :4] = scale_boxes(img[i].shape[1:], pred[:, :4], orig_imgs[i].shape[:2])
        results.append(results_mod.Results(orig_imgs[i], path=None, names=self.model.names, boxes=pred))
    return results

# Replace DetectionPredictor's postprocess method with the patched version
DetectionPredictor.postprocess = patched_postprocess