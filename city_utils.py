import torch
import torchvision.transforms as transforms
import numpy as np
import PIL
import city_config

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

def encode_segmap(mask, valid_classes, class_map):
    for class_idx in np.unique(mask):
        if class_idx not in valid_classes.keys():
            mask[mask == class_idx] = 0 # make the unwanted class as unlabelled
        
    for valid_idx in valid_classes.keys():
        mask[mask == valid_idx] = class_map[valid_idx] # correct the labels for valid classes

    return mask

def convert_input_images(img):
    img = img.resize((600, 300), resample=PIL.Image.NEAREST)
    img = trans(img)

    return img

def convert_input_masks(mask):
    mask = mask.resize((600, 300), resample=PIL.Image.NEAREST)
    mask = np.array(mask)
    mask = encode_segmap(mask, city_config.valid_classes, city_config.class_map)

    mask = torch.from_numpy(mask)
    mask = mask.type(torch.float32)

    return mask

