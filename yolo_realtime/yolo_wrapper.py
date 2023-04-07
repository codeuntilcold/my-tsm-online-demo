import time

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# CONFIGURATION
DEVICE = '0'
CHECKPOINT = './yolo_realtime/checkpoint/yolov7_tiny_phone_packaging_enhance_dataset/weights/best.pt'
TRACE = False
WEBCAM = True
DATA_SOURCE = '0'      # folder path for inference image or video, 0 for webcam
import numpy as np

class Yolo_Wrapper:  # multiple IP or RTSP cameras
    def __init__(self, checkpoint_path="", img_size=640):
        self.checkpoint_path = checkpoint_path
        self.img_size = img_size
        self.device = select_device(DEVICE)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(CHECKPOINT, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(640, s=self.stride)
        if TRACE:
            self.model = TracedModel(self.model, self.device, 640)
        if self.half:
            self.model.half()  # to FP16

        self.classify = False

        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()



        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1


    def detect(self, im0s, save_img=False):
        img = self.pre_process(im0s)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=False)[0]
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]
        t2 = time_synchronized()
        # Apply NMS
        print(pred)
        pred = non_max_suppression(pred, 0, 0.45, classes=None, agnostic=False)
        t3 = time_synchronized()
        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0s)
        # Process detections
        # print(pred)
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = '0', '%g: ' % i, im0s[i].copy()

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                #     if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)  # label format
                    print(('%g ' * len(line)).rstrip() % line)
                print('\n ------------------------------------ \n')

    def pre_process(self, img0):
        img = [self.letterbox(x, self.img_size, auto=True, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        return img


    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
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
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)