import numpy as np
import cv2
import torch
import torchvision
import onnxruntime as rt
from PIL import Image,ImageDraw,ImageFont
class YOLOV5_Detect():
    def __init__(self,img, ses):
        self.img = img.copy()
        self.img_size = img.shape[0:2]
        # self.model_path = modelPath
        # self.session = rt.InferenceSession(self.model_path)
        self.session = ses

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_area(self,box):
        # box = xyxy(4,n)
        return (box[2] - box[0]) * (box[3] - box[1])

    def box_iou(self, box1, box2, eps=1e-7):
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / (self.box_area(box1.T)[:, None] + self.box_area(box2.T) - inter + eps)
    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.3,
                            iou_thres=0.2,
                            agnostic = False,
                            max_det=300):
        bs = prediction.shape[0]  # batch size
        xc = prediction[..., 4] > conf_thres  # candidates
        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        merge = False  # use merge-NMS
        output = [torch.zeros((0, 6), device = prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
        return output

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self,boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def detect(self):
        img0 = self.img
        img = self.letterbox(img0, (640,640), stride=32, auto=False) #only pt use auto=True, but we are onnx
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(torch.device('cpu'))
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im = im.cpu().numpy()  # torch to numpy
        y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0] #inference onnx model to get the total output
        #non_max_suppression to remove redundant boxes
        y = torch.from_numpy(y).to(torch.device('cpu'))
        pred = self.non_max_suppression(y, conf_thres=0.35, iou_thres=0.3, agnostic=True, max_det=1000)
        #transform coordinate to original picutre size
        boxes_yolo = []
        boxes_loc = []
        index = []
        score = []
        for i, det in enumerate(pred):
            det[:, :4] = self.scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
            names = ['1']
            for *xyxy, conf, cls in reversed(det):
                con = round(float(conf), 4)
                c = int(cls)  # integer class
                (x1, x2) = (int(xyxy[0]), int(xyxy[2]))
                (y1, y2) = (int(xyxy[1]), int(xyxy[3]))
                center_x = (x1 + x2) / (2 * self.img_size[1])
                center_y = (y1 + y2) / (2 * self.img_size[0])
                w = abs((x2 - x1)) / self.img_size[1]
                h = abs((y2 - y1)) / self.img_size[0]
                box_yolo = ['{:.6f}'.format(center_x), '{:.6f}'.format(center_y), '{:.6f}'.format(w), '{:.6f}'.format(h)]
                box_loc = [x1, y1, x2, y2]
                score.append(con)
                index.append(c)
                boxes_yolo.append(box_yolo)
                boxes_loc.append(box_loc)
        return boxes_loc, boxes_yolo, index, score

if __name__ == '__main__':
    model = rt.InferenceSession('./best.onnx')
    img = cv2.imread('18.jpg')
    height, width = img.shape[:2]
    new_height = height * 4
    new_width = width * 4
    # 调用resize函数进行放大
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    yolo = YOLOV5_Detect(resized_img, model)
    boxes_loc, boxes_yolo, index, score = yolo.detect()
    print(score)
    # 在图像上绘制边界框
    for box, class_id, score in zip(boxes_loc, index, score):
        x1, y1, x2, y2 = box
        cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_img, 'class: ' + str(class_id), (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 1)

    # 保存结果图像
    cv2.imshow('img',resized_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

