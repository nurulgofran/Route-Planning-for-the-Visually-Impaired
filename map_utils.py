import json
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data

def convert_input_images(img):
    img = img.resize((600, 300), resample=PIL.Image.NEAREST)
    img = trans(img)

    return img

def convert_input_masks(mask):
    mask = mask.resize((600, 300), resample=PIL.Image.NEAREST)
    mask = np.array(mask) 
    mask = torch.from_numpy(mask)
    mask = mask.type(torch.float32)

    return mask

def put_colors_in_grayscale_mask(grayscale_mask, n_classes, label_colors):
    #convert grayscale to color
    if not isinstance(grayscale_mask,np.ndarray):
        temp = grayscale_mask.numpy()
    else:
        temp = grayscale_mask
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colors[l][0]
        g[temp == l] = label_colors[l][1]
        b[temp == l] = label_colors[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb


