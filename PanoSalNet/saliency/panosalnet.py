#TODO: load the panosalnet model, create salient map from input image
#input: model config file: panosalnet_test.prototxt
#       model weight: 'panosalnet_iter_800.caffemodel
#       input image: test.png
#output: salient map

import cv2
import numpy as np
# import caffemodel2pytorch as caffe
import timeit
import os
import h5py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, 'panosalnet_test.prototxt')
os.chdir(script_directory)

# FILE_MODEL_CONFIG = 'panosalnet_test.prototxt'
# FILE_MODEL_WEIGHT = 'panosalnet_iter_800.caffemodel'
# FILE_IMAGE = 'test.png'


class PanoSalNet(nn.Module):
    def __init__(self):
        super(PanoSalNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(512, 256, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(256, 128, kernel_size=11, padding=5)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 32, kernel_size=11, padding=5)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(32, 1, kernel_size=13, padding=6)

        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=8, stride=4, padding=2, bias=False)

        # Note: EuclideanLoss is not needed in the forward pass for inference, as it's typically used during training.

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.relu8(x)

        x = self.conv9(x)

        x = self.deconv1(x)

        return x

def post_filter(_img):
    result = np.copy(_img)
    result[:3,:3] = _img.min()
    result[:3, -3:] = _img.min()
    result[-3:, :3] = _img.min()
    result[-3:, -3:] = _img.min()
    return result

def generate_salient_image(model, img_path, output_folder):
    # H, W = 288, 512
    # H, W = 216, 384
    H, W = 256, 512
    img = cv2.imread(img_path)
    dat = cv2.resize(img, (W, H))
    dat = dat.astype(np.float32)
    dat -= np.array(mu) # mean values across training set
    dat *= input_scale
    dat = dat.transpose((2,0,1))

    # Convert the data to torch tensor and add a batch dimension
    dat_tensor = torch.from_numpy(dat).unsqueeze(0)

    # Forward pass
    output = model(dat_tensor)

    # Get the salient map from the output
    salient = output[0][0].detach().numpy()
    salient = (salient * 1.0 - salient.min())
    salient = (salient / salient.max()) * 255
    salient = post_filter(salient)
    salient_image = salient.squeeze().astype(np.uint8)
    colored_salient_image = cv2.applyColorMap(salient_image, cv2.COLORMAP_VIRIDIS)
    # save colored salient image
    vid_number, img_number = img_path.split('/')[-2:]
    rgb_path = os.path.join(output_folder, 'rgb', vid_number, img_number)
    cv2.imwrite(rgb_path, colored_salient_image)
    # save black and white image
    bw_path = os.path.join(output_folder, 'bw', vid_number, img_number)
    cv2.imwrite(bw_path, salient_image)

def generate_salient_images_in_directory(model, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Traverse through all images in the input folder
    for subdir, _, files in os.walk(input_folder):
        files = sorted([f for f in files if f.endswith('.png')])
        for i in range(0, len(files)): #, 8):
            file = files[i]
            # Construct the full path to the image
            img_path = os.path.join(subdir, file)

            vid_number = input_folder.split('/')[-1]
            rgb_subfolder = os.path.join(output_folder, 'rgb', vid_number)
            bw_subfolder = os.path.join(output_folder, 'bw', vid_number)
            
            os.makedirs(rgb_subfolder, exist_ok=True)
            os.makedirs(bw_subfolder, exist_ok=True)

            # Generate salient image
            print(f'Generating saliency map for: {img_path}')
            generate_salient_image(model, img_path, output_folder)

# Load dumped Panosalnet weights in PyTorch
panosalnet_model = PanoSalNet()
state_dict = torch.load('panosalnet_iter_800.caffemodel.pt')
panosalnet_model.load_state_dict(state_dict=state_dict)
# Set the model to evaluation mode
panosalnet_model.eval()

# mu = [104.00698793, 116.66876762, 122.67891434]
mu = [ 110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0 ]
std = [ 38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0 ]

input_scale = 1.0 / 128

input_folder = '../../dataset/dsav360/360videos_with_ambisonic/processed/test/0018'
# input_folder = 'test_input'
output_folder = 'output_saliency_maps'
generate_salient_images_in_directory(panosalnet_model, input_folder, output_folder)