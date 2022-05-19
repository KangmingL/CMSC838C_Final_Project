from torch.utils.data import Dataset
import cv2
from faster_rcnn import *
from skimage.io import imread, imsave
import numpy as np
from skimage.transform import estimate_transform, warp, resize, rescale
import os.path as osp

class ImageDataset(Dataset):
    def __init__ (self, data, data_path, mode, scale=1.1, crop_size=224, device='cuda:0'):
        super(ImageDataset, self).__init__()
        self.mode = mode
        self.data = data[mode]
        self.img_path = self.data['img_fns']
        self.pose = self.data['pose']
        self.keypoints = self.data['keypoints2D']
        self.length = len(self.data['img_fns'])
        self.detector = FasterRCNN(device=device)
        self.crop_size = crop_size
        self.scale = scale
        self.data_path = data_path
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.img_path[idx]=self.img_path[idx].replace('png','jpg')
        img_path = osp.join(self.data_path, self.img_path[idx])
        img = imread(img_path)/255.
        h, w, _ = img.shape
        image_tensor = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)[None, ...]
        bbox = self.detector.run(image_tensor)
        if bbox is None:
            print('no person detected! run original image')
            left = 0; right = w-1; top=0; bottom=h-1
        else:
            left = bbox[0]; right = bbox[2]; top = bbox[1]; bottom = bbox[3]
        old_size = max(right - left, bottom - top)
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size*self.scale)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        dst_image = warp(img, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        dst_image = dst_image.transpose(2,0,1)
        # d_h, d_w = 640 - min(640, h), 640 - min(640, w)
        # top, bottom = d_h//2, d_h - (d_h//2)
        # left, right = d_w//2, d_w - (d_w//2)
        # pad images to size 640x640
        # img = cv2.copyMakeBoarder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return {'image': torch.tensor(dst_image).float(), 
                'pose':torch.tensor(self.pose[idx,1:22,:].flatten()).float(), 
                'keypoints':torch.tensor(self.keypoints[idx,:,:].flatten()).float(),
                'image_path':self.img_path[idx]}



