import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
 # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import click
import cv2
from PIL import Image
def global_linear_transmation(img):
    maxV = img.max()
    minV = img.min()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = ((img[i, j] - minV) * 255) / (maxV - minV)
    return img
def depth2color(depth):
    depth = np.array(depth, np.uint8)
    color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return color
def test_simple(model, img_path):
    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()

    img = np.float32(io.imread(img_path))/255.0
    img = resize(img, (input_height, input_width), order = 1)
    input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
    input_img = input_img.unsqueeze(0)

    input_images = Variable(input_img.cuda() )
    pred_log_depth = model.netG.forward(input_images) 
    pred_log_depth = torch.squeeze(pred_log_depth)

    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)
    pred_inv_depth = global_linear_transmation(pred_inv_depth)

    # pred_inv_depth = np.expand_dims(pred_inv_depth, axis = 2)
    # pred_inv_depth = np.concatenate((pred_inv_depth, pred_inv_depth, pred_inv_depth), axis = -1)
    # image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    color = depth2color(pred_inv_depth)

    # image_rgb = cv2.cvtColor(pred_inv_depth, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(img_path.split('.')[0] + '.png', color)
    # print(pred_inv_depth.shape)
    sys.exit()


opt = TrainOptions().parse()
model, img_path = create_model(opt)

input_height = 384
input_width  = 512

test_simple(model, img_path)
print("We are done")
