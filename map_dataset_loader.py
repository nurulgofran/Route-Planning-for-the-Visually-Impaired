import os
from PIL import Image
from torch.utils.data import Dataset

class Mapillary(Dataset):
    def __init__(self, data_path, transform, target_transform):
        self.data_path = data_path
        self.img_names = sorted(os.listdir(os.path.join(data_path, "images")))
        self.mask_names = sorted(os.listdir(os.path.join(data_path, "grayscale_labels")))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
      # load image
      img_name = self.img_names[idx]
      img_path = os.path.join(self.data_path, "images", img_name)
      img = Image.open(img_path)
      image = self.transform(img)

      # load mask
      mask_name = self.mask_names[idx]
      mask_path = os.path.join(self.data_path, "grayscale_labels", mask_name)
      mask = Image.open(mask_path) # read grayscale
      semantic_mask = self.target_transform(mask)

      return image, semantic_mask
    
    def __len__(self):
        return len(self.img_names)

