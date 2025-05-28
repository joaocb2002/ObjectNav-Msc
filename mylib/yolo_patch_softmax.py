# yolo_patch.py

import ultralytics.engine.results
import ultralytics.utils.ops
import torch
from ultralytics.utils.ops import xywh2xyxy, LOGGER, nms_rotated
import ultralytics.engine.results as results_mod
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils.ops import scale_boxes, convert_torch2numpy_batch
import torchvision

# Temperature for softmax scaling
TEMPERATURE = 1.0

OriginalBoxes = results_mod.Boxes

def init(self, boxes, orig_shape) -> None:
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    n = boxes.shape[-1]

    OriginalBoxes.__init__(self, boxes, orig_shape)

    self.orig_shape = orig_shape
    self.is_track = False
    self.num_classes = 0

    if n == 6:
        self.format = 'xyxy_conf_cls'
    elif n == 7:
        self.format = 'xyxy_conf_cls_track'
        self.is_track = True
    else:
        self.format = 'xyxy_conf_cls_classconf'
        self.num_classes = n - 6

    self.data = boxes

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

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    prediction = prediction.transpose(-1, -2)

    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)

    output = [torch.zeros((0, 6 + nc + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        box, cls_conf, mask = x.split((4, nc, nm), 1)

        # Clamp sigmoid outputs to avoid numerical instability
        cls_conf = cls_conf.clamp(min=1e-6, max=1 - 1e-6)

        # Compute logits
        logits = torch.log(cls_conf / (1.0 - cls_conf))

        # Apply temperature-scaled softmax
        softmax = torch.nn.functional.softmax(logits / TEMPERATURE, dim=1)

        # Use top softmax class and confidence as summary
        conf, j = softmax.max(1, keepdim=True)

        # Construct full output vector
        x = torch.cat((box, conf, j.float(), softmax, mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        n = x.shape[0]
        if not n:
            continue

        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]

        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c
            i = torchvision.ops.nms(boxes, scores, iou_thres)

        i = i[:max_det]
        output[xi] = x[i]

        if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 1e9:
            LOGGER.warning("Memory usage exceeded during NMS")

    return output

ultralytics.utils.ops.non_max_suppression = non_max_suppression

original_results_init = results_mod.Results.__init__

def patched_results_init(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None):
    original_results_init(self, orig_img, path, names, boxes, masks, probs, keypoints, obb)

results_mod.Results.__init__ = patched_results_init

original_postprocess = DetectionPredictor.postprocess

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

DetectionPredictor.postprocess = patched_postprocess

class PatchedBoxes(OriginalBoxes):
    def __init__(self, boxes, orig_shape):
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]

        self.orig_shape = orig_shape
        self.is_track = False
        self.num_classes = 0

        if n == 6:
            self.format = 'xyxy_conf_cls'
        elif n == 7:
            self.format = 'xyxy_conf_cls_track'
            self.is_track = True
        else:
            self.format = 'xyxy_conf_cls_classconf'
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

results_mod.Boxes = PatchedBoxes
