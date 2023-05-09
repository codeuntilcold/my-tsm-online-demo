from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch

def preprocess_yolo_output(yolo_ouput):
    # fix_label_dict = {
    #     0: 0,
    #     1:4,
    #     2:3,
    #     3:2,
    #     4:1
    # }
    result = torch.zeros(20)
    map_label_prob = {}
    for ele in yolo_ouput:
        label = int(ele[0].item())
        # label = fix_label_dict[label]
        coordinate = ele[1:-1]
        prob = ele[-1].item()
        # if label == 3:
        #     continue
        if label in map_label_prob.keys():
            if prob < map_label_prob[label]:
                continue
        map_label_prob[label] = prob
        # Shift label larger 3 backward (cause remove label 3)
        # label = label - 1 if label > 3 else label
        result[label*4:(label+1)*4] = torch.tensor(coordinate)
    return result

class AOD(Module):

    def __init__(self, numChannels, classes):

        super(AOD, self).__init__()
        self.conv1 = Conv1d(in_channels=numChannels, out_channels=32,kernel_size=(3))
        self.relu1 = ReLU()
		# self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = Conv1d(in_channels=32, out_channels=64,kernel_size=(3))
        self.relu2 = ReLU()

		# self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
        # self.conv3 = Conv1d(in_channels=32, out_channels=64, kernel_size=(3))
		# self.relu3 = ReLU()
        self.fc1 = Linear(in_features=4*64, out_features=64)
        self.relu4 = ReLU()
		# initialize our softmax classifier
        self.fc2 = Linear(in_features=64, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
		# x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
		# x = self.maxpool2(x)

        x = flatten(x,1)
        x = self.fc1(x)
        x = self.relu4(x)
		# predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
		# return the output predictions
        return output