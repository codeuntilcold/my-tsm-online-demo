import numpy as np
import cv2
import time
import torch
import torchvision
import os
from PIL import Image
from ops.models import TSN
from tsm_mobilenet_v2 import MobileNetV2
import matplotlib.pyplot as plt

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True


# Load the MobileNet weights, return an executor and context
def get_executor(use_gpu=True):
    torch_module = MobileNetV2(n_class=27)
    # checkpoint not downloaded
    checkpoint = "mobilenetv2_jester_online.pth.tar"
    # checkpoint = "TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth"
    if not os.path.exists(checkpoint):
        print('Downloading PyTorch checkpoint...')
        import urllib.request
        url = f'https://hanlab.mit.edu/projects/tsm/models/{checkpoint}'
        urllib.request.urlretrieve(url, f'./{checkpoint}')
    torch_module.load_state_dict(torch.load(checkpoint))

    # torch_module.features[0].register_forward_hook(get_activation('layer0'))
    # torch_module.features[6].register_forward_hook(get_activation('layer6'))
    # torch_module.features[12].register_forward_hook(get_activation('layer12'))
    # torch_module.features[11].register_forward_hook(get_activation('layer18'))

    if use_gpu:
        target = 'cuda'
    else:
        target = 'llvm -mcpu=cortex-a72 -target=armv7l-linux-gnueabihf'

    torch_module.eval()
    with torch.no_grad():
        def executor(input, buffer):
            return torch_module(input, *buffer)
        return executor, target


def get_phonepackaging_executor():
    torch_module = TSN(num_class=len(categories), 
                        num_segments=25, 
                        modality='rgb',
                        base_model='resnet50',
                        is_shift=True,
                        )
    weights = "TSM_PhonePackaging_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth.tar"
    checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    checkpoint = checkpoint['state_dict']
    sd = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in sd:
            sd[v] = sd.pop(k)
    torch_module.load_state_dict(sd)
    torch_module.eval()
    with torch.no_grad():
        return torch_module, "empty target..."


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame


def get_transform():
    class GroupScale(object):
        """ Rescales the input PIL.Image to the given 'size'.
        'size' will be the size of the smaller edge.
        For example, if height > width, then image will be
        rescaled to (size * height / width, size)
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
        """

        def __init__(self, size, interpolation=Image.BILINEAR):
            self.worker = torchvision.transforms.Resize(size, interpolation)

        def __call__(self, img_group):
            return [self.worker(img) for img in img_group]

    class GroupCenterCrop(object):
        def __init__(self, size):
            self.worker = torchvision.transforms.CenterCrop(size)

        def __call__(self, img_group):
            return [self.worker(img) for img in img_group]

    class Stack(object):

        def __init__(self, roll=False):
            self.roll = roll

        def __call__(self, img_group):
            if img_group[0].mode == 'L':
                return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
            elif img_group[0].mode == 'RGB':
                if self.roll:
                    return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
                else:
                    return np.concatenate(img_group, axis=2)

    class ToTorchFormatTensor(object):
        """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
        to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

        def __init__(self, div=True):
            self.div = div

        def __call__(self, pic):
            if isinstance(pic, np.ndarray):
                # handle numpy array
                img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
            else:
                # handle PIL Image
                img = torch.ByteTensor(
                    torch.ByteStorage.from_buffer(pic.tobytes()))
                img = img.view(pic.size[1], pic.size[0], len(pic.mode))
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
            return img.float().div(255) if self.div else img.float()

    class GroupNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
            rep_std = self.std * (tensor.size()[0] // len(self.std))

            # TODO: make efficient
            for t, m, s in zip(tensor, rep_mean, rep_std):
                t.sub_(m).div_(s)

            return tensor

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
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]): #  and history[-2] == history[-3]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


# WE DON'T HAVE A LABEL FOR IDLE ACTIONS
categories = [
    "pick up phone box",                        # 0
    "open phone box",                           # 1
    "put down phone box ",                      # 2
    "put down phone box cover",                 # 3
    "pick up phone from phone box",             # 4
    "put down phone on table",                  # 5
    "pick up instruction paper from phone box",  # 6
    "put down instruction paper on table",      # 7
    "pick up earphones from phone box",         # 8
    "put down earphones on table",              # 9
    "pick up charger from phone box",           # 10
    "put down charger on table",                # 11
    "pick up charger from table",               # 12
    "put down charger into phone box",          # 13
    "pick up earphones from table",             # 14
    "put down earphones into phone box",        # 15
    "pick up instruction paper from table",     # 16
    "put down instruction paper into phone box",# 17
    "pick up phone from table",                 # 18
    "inspect phone",                            # 19
    "put down phone into phone box",            # 20
    "pick up phone box cover",                  # 21
    "close phone box",                          # 22
]

# categories_mobilenetv2 = [
#     "Doing other things",  # 0
#     "Drumming Fingers",  # 1
#     "No gesture",  # 2
#     "Pulling Hand In",  # 3
#     "Pulling Two Fingers In",  # 4
#     "Pushing Hand Away",  # 5
#     "Pushing Two Fingers Away",  # 6
#     "Rolling Hand Backward",  # 7
#     "Rolling Hand Forward",  # 8
#     "Shaking Hand",  # 9
#     "Sliding Two Fingers Down",  # 10
#     "Sliding Two Fingers Left",  # 11
#     "Sliding Two Fingers Right",  # 12
#     "Sliding Two Fingers Up",  # 13
#     "Stop Sign",  # 14
#     "Swiping Down",  # 15
#     "Swiping Left",  # 16
#     "Swiping Right",  # 17
#     "Swiping Up",  # 18
#     "Thumb Down",  # 19
#     "Thumb Up",  # 20
#     "Turning Hand Clockwise",  # 21
#     "Turning Hand Counterclockwise",  # 22
#     "Zooming In With Full Hand",  # 23
#     "Zooming In With Two Fingers",  # 24
#     "Zooming Out With Full Hand",  # 25
#     "Zooming Out With Two Fingers"  # 26
# ]

# For visualizing activation of layers
activation_hist = {
    "layer0": [],
    "layer6": [],
    "layer12": [],
    "layer18": [],
}

def get_activation(name):
    def hook(model, input, output):
        activation_hist[name].append(output)
    return hook


def main():
    print("Open camera...")
    cap = cv2.VideoCapture(0)

    print(cap)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    WINDOW_NAME = 'Video Gesture Recognition'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    t = None
    index = 0
    print("Build transformer...")
    transform = get_transform()
    print("Build Executor...")
    # executor, ctx = get_executor()
    executor, ctx = get_phonepackaging_executor()

    # Specific for MobileNetV2?
    buffer = [torch.zeros([1, 3, 56, 56]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 20, 7, 7]),
              torch.zeros([1, 20, 7, 7])]

    idx = 0
    history = [2]
    history_logit = []

    i_frame = -1

    print("Ready!")
    while True:
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        # Random frame for testing
        # img = np.random.randint(255, size=(240, 320, 3), dtype=np.uint8)
        if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
            t1 = time.time()
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            input_var = torch.autograd.Variable(
                img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))

            img_nd = input_var.numpy()
            outputs = executor(torch.from_numpy(img_nd), buffer)
            feat, buffer = outputs[0], outputs[1:]

            feat = feat.detach().numpy()
            if SOFTMAX_THRES > 0:
                feat_np = feat.reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                print(max(softmax))
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat, axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat, axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat)
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)

            t2 = time.time()
            # print(f"{index} {categories[idx]}")

            current_time = t2 - t1

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + categories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.5f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()


main()


def save_activations():
    print(f"History len: {len(activation_array)}")

    act = activation_hist['layer0'][0][0][0]
    print(f"Number of kernels: {act.size(0)}")
    print(f"Figsize: {act.size(1)}")


    act = activation_hist['layer6'][0][0][0]
    print(f"Number of kernels: {act.size(0)}")
    print(f"Figsize: {act.size(1)}")

    act = activation_hist['layer12'][0][0][0]
    print(f"Number of kernels: {act.size(0)}")
    print(f"Figsize: {act.size(1)}")

    act = activation_hist['layer18'][0][0][0]
    print(f"Number of kernels: {act.size(0)}")
    print(f"Figsize: {act.size(1)}")

    for layer_name, activation_array in activation_hist.items():
        for idx, activation_at_time in enumerate(activation_array):
            acti = activation_at_time[0][0]
            hw = (8, int(acti.size(0)/8))
            width = max(min(hw), 1)
            height = max(hw)
            fig, axarr = plt.subplots(width, height, sharex=True, sharey=True, figsize=(12,8))
            axarr = axarr.ravel()
            for i in range(acti.size(0)):
                axarr[i].imshow(acti[i].detach())
                os.makedirs(f"temp1/{layer_name}", exist_ok=True)
                fig.savefig(f"temp1/{layer_name}/fig{idx}.png")
            plt.close()
