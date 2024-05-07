import os
import sys
import cv2
import numpy as np
import map_utils
import map_config

rgb_mask_mappings = map_utils.load_json(map_config.json_path)
class_color_map = {} # will store the key:values as valid_class_idx:color in rgb mask

# build the class_color_map using valid_classes and rgb mask
for count, valid_class in enumerate(map_config.valid_classes):
    for item in rgb_mask_mappings["labels"]:
        if item["readable"].lower() == valid_class:
            class_color_map[count] = item["color"]

# construct the grayscale mask by replacing valid_class colors to valid_class_idx using class_color_map
print("\nconverting rgb masks to grayscale...\n")

if not os.path.exists(map_config.op_path):
    print("\noutput directory does not exist, please check that op_path in map_config.py exists\n")
    sys.exit()

for count, img_name in enumerate(sorted(os.listdir(map_config.masks_path))):
    img_path = os.path.join(map_config.masks_path, img_name)
    img = cv2.imread(img_path)
    rgb_mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = rgb_mask.shape
    grayscale_mask = np.zeros((h, w))
    
    for class_id, color in class_color_map.items():
        class_pixel_pos = np.all((rgb_mask == color), axis=-1)
        grayscale_mask[class_pixel_pos] = class_id

    # Save the grayscale image
    output_path = os.path.join(map_config.op_path, img_name)
    cv2.imwrite(output_path, grayscale_mask)

    if count%5 == 0:
        print(f"\n{count} masks done\n")

print("\nmasks conversion completed\n")




