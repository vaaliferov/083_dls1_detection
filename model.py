import numpy as np
import onnxruntime

from PIL import Image, ImageDraw
from PIL import ImageFont, ImageOps

FONT_SIZE = 15
FONT_PATH = 'fonts/DejaVuSans.ttf'

COLORS = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', 
          '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB', 
          '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', 
          '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')

def hex2rgb(h):  # rgb order (PIL)
    return tuple(int(h[1+i:1+i+2], 16) for i in (0,2,4))

def get_color(i):
    return hex2rgb('#' + COLORS[int(i) % len(COLORS)])

def load_font(font_path, font_size):
    return ImageFont.truetype(font_path, font_size)

def load_model(path):
    return onnxruntime.InferenceSession(path)

def load_labels(path):
    return np.array(open(path).read().splitlines())

def pad_image(im):
    w, h = im.size; m = np.max([w, h])
    hp, hpr = (m - w) // 2, (m - w) % 2
    vp, vpr = (m - h) // 2, (m - h) % 2
    return (hp + hpr, vp + vpr, hp, vp)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def clip_coords(boxes, shape):  
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def iou(box, boxes, box_area, boxes_area):
    assert boxes.shape[0] == boxes_area.shape[0]
    ys1 = np.maximum(box[0], boxes[:, 0])
    xs1 = np.maximum(box[1], boxes[:, 1])
    ys2 = np.minimum(box[2], boxes[:, 2])
    xs2 = np.minimum(box[3], boxes[:, 3])
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    unions = box_area + boxes_area - intersections
    ious = intersections / unions
    return ious

def nms(boxes, scores, threshold):
    assert boxes.shape[0] == scores.shape[0]
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes): break
        ious = iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index).astype(int)

def non_max_suppression(prediction, conf_thres, iou_thres):

    max_box_wh = 4096
    max_boxes_nms = 1000
    
    xc = prediction[..., 4] > conf_thres # candidates
    output = [np.zeros((0, 6))] * prediction.shape[0]
    
    for xi, x in enumerate(prediction): # image index, image inference
        
        x = x[xc[xi]] # confidence
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])
        
        # best class only
        conf = np.max(x[:, 5:], axis=1)
        conf = np.expand_dims(conf, 1)
        j = np.argmax(x[:, 5:], axis=1)
        j = np.expand_dims(j, 1)
        
        x = np.concatenate((box, conf, j.astype('float32')), 1)
        x = x[conf.reshape(-1) > conf_thres]
        
        x = x[-x[:, 4].argsort()[:max_boxes_nms]]  # sort by confidence
        
        c = x[:, 5:6] * max_box_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        output[xi] = x[nms(boxes, scores, iou_thres)]
        
    return output

def draw_one_box(draw, box, color, label, font, margin=4, width=2):
    _, _, w, h = font.getbbox(label)
    text_box = (box[0] + margin, box[1] + margin)
    plate_box = (box[0], box[1], box[0] + w + margin * 2, box[1] + h + margin * 2)
    draw.rectangle(box, fill=None, outline=color, width=width)
    draw.rectangle(plate_box, fill=color, outline=None)
    draw.text(text_box, label, font=font, fill=hex2rgb('#FFFFFF'))


class Model:

    def __init__(self, model_path, labels_path):
        
        self.model = load_model(model_path)
        self.labels = load_labels(labels_path)
        self.font = load_font(FONT_PATH, FONT_SIZE)

    def detect(self, path, conf_thres=0.25, iou_thres=0.45):

        image = Image.open(path)

        im = image.copy()
        im = ImageOps.exif_transpose(im)
        im = ImageOps.expand(im, pad_image(im))
        im.thumbnail((640,640), Image.LANCZOS)

        x = np.float32(np.array(im) / 255.)
        
        x = x.transpose(2,0,1)
        x = x.reshape((1,) + x.shape)
        
        outs = self.model.run(None, {self.model.get_inputs()[0].name: x})
        
        det = non_max_suppression(outs[0], conf_thres, iou_thres)[0]
        det[:,:4] = scale_coords((640,640), det[:,:4], image.size[::-1]).round()

        draw = ImageDraw.Draw(image)

        for *box, conf, cls in det:
            color = get_color(int(cls))
            label = (f'{self.labels[int(cls)]} {conf:.2f}')
            draw_one_box(draw, box, color, label, self.font)

        image.save(path)