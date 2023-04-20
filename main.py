import sys
sys.path.append('./yolo_realtime')

from cv2 import CAP_DSHOW
import numpy as np
import cv2
import time
import torch
import torchvision
from PIL import Image
from ops.models import TSN
from ops.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize 
from numpy.random import randint
from yolo_realtime.yolo_wrapper import *
from action_transition_graph.graph_connector import GraphConnector
import argparse
import os

parser = argparse.ArgumentParser(description='TSM for phone packaging recognition')
parser.add_argument('-c','--class', default=11, help='Number of action class', required=False, type=int)
parser.add_argument('-r','--refine', help='Refine output by smoothing history', required=False)
parser.add_argument('-l','--logit', help='Refine output by averaging', required=False)
parser.add_argument('-t','--thres', help='Softmax threshold', default=0.1, type=float)
parser.add_argument('-w','--webcam', help='Using external webcam', default=False)
parser.add_argument('-i','--idcam', help='ID of camera', default=1, type=int)
parser.add_argument('-k','--topk', help='Print top-k', default=1, type=int)
parser.add_argument('-p','--path', help='Model path id', default=1, type=int)
parser.add_argument('-v','--use-video', help='Use video in this path instead of camera',
                    default='none', type=str)
args = vars(parser.parse_args())

DEVICE = 'cuda'
NUM_CLASS = args['class']
NUM_SEGMENTS = 8
TOP_K = args['topk']
SOFTMAX_THRES = args['thres']
HISTORY_LOGIT = args['logit']
REFINE_OUTPUT = args['refine']
LOGITECH_CAM = args['webcam']
CAMERA_ID = args['idcam']
USE_VIDEO_PATH = args['use-video']
CATEGORIES = [
    "open phone box",
    "take out phone",
    "take out instruction paper",
    "take out earphones",
    "take out charger",
    "put in charger",
    "put in earphones",
    "put in instruction paper",
    "inspect phone",
    "put in phone",
    "close phone box",
    "no action"
]
PATHS = [
    'checkpoints/TSM_PhonePackaging_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt_sliding.best.pth.tar', #0
    'checkpoints/no_action_shift_8/ckpt.best.pth.tar', #1
    'checkpoints/extradata_slidingwindow32+64_shift8/ckpt.best.pth.tar', #2
    'checkpoints/extradata_slidingwindow32_shift8/ckpt.best.pth.tar', #3
    'checkpoints/noaction_extradata_slidingwindow32+64_shift8/ckpt.best.pth.tar' #4
]


def get_executor():
    path_pretrain = PATHS[args['path']]
    # is_shift, shift_div, shift_place = parse_shift_option_from_log_name(path_pretrain)
    # base_model = path_pretrain.split('TSM_')[1].split('_')[2]
    is_shift, shift_div, shift_place = True, 8, "blockres"
    base_model = 'resnet50'
    torch_module = TSN(NUM_CLASS, NUM_SEGMENTS, 'RGB',
                base_model=base_model,
                consensus_type='avg',
                img_feature_dim=256,
                pretrain='imagenet',
                is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
                non_local='_nl' in path_pretrain)

    if path_pretrain:
        print(f"=> Load pretrain model from '{path_pretrain}'")
        checkpoint = torch.load(path_pretrain, map_location=torch.device(DEVICE))
        checkpoint = checkpoint['state_dict']

        sd = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                        'base_model.classifier.bias': 'new_fc.bias',
                        }
        for k, v in replace_dict.items():
            if k in sd:
                sd[v] = sd.pop(k)

        torch_module.load_state_dict(sd)
        torch_module.to(DEVICE)
        torch_module.eval()

    with torch.no_grad():
        return torch_module


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame


def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT or len(history) < 3:
        return idx_, history

    max_hist_len = 32  # max history buffer

    # history smoothing
    if idx_ != history[-1]:
        if history[-1] != history[-2]:
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


def get_offset_segment(num_frames):
    new_length = 1
    num_segments = 8
    average_duration = (num_frames - new_length + 1) // num_segments
    if average_duration > 0:
        offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration, size=num_segments)
    elif num_frames > num_segments:
        offsets = np.sort(randint(num_frames - new_length + 1, size=num_segments))
    else:
        offsets = np.zeros((num_segments,))

    return offsets


def stream_frames():
    if USE_VIDEO_PATH != 'none':
        #### Read frames from images
        all_images = []
        for img_name in os.listdir(USE_VIDEO_PATH):
            img = cv2.imread(os.path.join(USE_VIDEO_PATH, img_name))
            all_images.append(img)
        for img in all_images:
            yield True, img
    else:
        print("Open camera...")
        # set a lower resolution for speed up
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        if LOGITECH_CAM:
            cap = cv2.VideoCapture(CAMERA_ID, apiPreference=CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(CAMERA_ID)
        print(cap)
        while True:
            yield cap.read()
        cap.release()


def main():
    frames = stream_frames()

    # env variables
    full_screen = False
    WINDOW_NAME = 'Phone Packaging Inspection'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 640)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    print("Build transformer...")
    transform_crop = get_transform()
    print("Build Executor...")
    executor = get_executor()

    idx_no_action = len(CATEGORIES)
    buffer = torch.tensor([])
    history = [2]
    history_lens = [32]
    history_logit = []

    # yolov7 = Yolo_Wrapper()
    conn = GraphConnector()

    print("Ready!")
    while True:
        ret, img = next(frames)  # (480, 640, 3) 0 ~ 255
        # with torch.no_grad():
        #     yolov7.detect([img])
        
        if not ret:
            print("failed to grab frame")
            continue
        
        t1 = time.time()

        img_resize = cv2.resize(img, (256, 256))
        img_tran = transform_crop([Image.fromarray(img_resize).convert('RGB')])
        input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))

        # Save input up to present
        buffer = torch.cat((buffer, input_var))
        buffer = buffer[-64:]
        
        # For each history lengths, get sampled input vectors
        inputs = []
        for w in history_lens:
            window = buffer[-w:]
            offset = get_offset_segment(len(window))
            inputs.append(window[offset])
        
        # For each input, execute and get output feature
        feats = []
        for inp in inputs:
            inp = inp.to(DEVICE)
            feats.append(executor(inp))
            inp.detach()

        # For each output feature, calculate softmax
        softmaxes = []
        for feat in feats:
            feat = feat.detach().numpy() if DEVICE == 'cpu' \
                else feat.detach().cpu().numpy()
            
            feat_np = feat.reshape(-1)
            feat_np -= feat_np.max()
            softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
            softmaxes.append(softmax)
            idx_ = np.argmax(feat, axis=1)[0] if max(softmax) > SOFTMAX_THRES else idx_no_action

        # Weight by history length/window
        weight_per_window = np.array([[1]*NUM_CLASS, [1]*NUM_CLASS])
        softmaxes = np.sum(softmaxes * weight_per_window, axis=0)
        idx_s = np.argsort(softmaxes)[-TOP_K:]
        
        if HISTORY_LOGIT:
            history_logit.append(feat)
            history_logit = history_logit[-12:]
            avg_logit = sum(history_logit)
            idx_ = np.argmax(avg_logit, axis=1)[0]

        if REFINE_OUTPUT:
            idx_, history = process_output(idx_, history)

        t2 = time.time()
        exec_time = t2 - t1

        # img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        whiteboard = np.zeros([height // 2, width, 3]).astype('uint8') + 255

        for i, idx_ in enumerate(idx_s[::-1]):
            acc = softmax[idx_]
            
            if acc > SOFTMAX_THRES:
                conn.send(f"{idx_} {acc}")

            cv2.putText(whiteboard, f"{CATEGORIES[idx_] if acc>SOFTMAX_THRES else '-'}",
                        (8, int(height / 16) + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
            
            cv2.putText(whiteboard, f"{acc:.2f}",
                        (width - 170, int(height / 16) + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
        
        # cv2.putText(whiteboard, f"FPS: {1.0 / exec_time:2f}",
        #             (8, int(height) / 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, (0, 0, 0), 2)

        img = np.concatenate((img, whiteboard), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Toggle full screen option!')
            full_screen = not full_screen
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, 
                                  cv2.WINDOW_FULLSCREEN if full_screen else cv2.WINDOW_NORMAL)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Load the MobileNet weights, return an executor and context
# def parse_shift_option_from_log_name(log_name):
#     if 'shift' in log_name:
#         strings = log_name.split('_')
#         for i, s in enumerate(strings):
#             if 'shift' in s:
#                 break
#         return True, int(strings[i].replace('shift', '')), strings[i + 1]
#     else:
#         return False, None, None

##### Load images
# for image_index in range(len(all_images)):
    # img = all_images[image_index]        
##### Write image to folder
# cv2.imwrite(f"4_1_1\\{image_index:010d}.png", img)
##### Show transformed image
# cv2.imshow(WINDOW_NAME, img_tran.permute(1, 2, 0).numpy())
