import os
import sys
import torch
import numpy as np
from PIL import Image
import map_utils
import map_config
import matplotlib.pyplot as plt
from city_training import cityscape_deeplab
from training_pipeline import mapillary_deeplab

if not os.path.exists(map_config.op_dir):
    print(f'op_dir i.e. {map_config.op_dir} does not exists. Please create one')
    sys.exit()

if torch.cuda.is_available():
    checkpoint = torch.load(map_config.map_ckp_path)
else:
    checkpoint = torch.load(map_config.map_ckp_path, map_location=torch.device('cpu'))

cityscape_model = cityscape_deeplab(map_config.n_classes)
model = mapillary_deeplab(map_config.n_classes, cityscape_model)
model.load_state_dict(checkpoint["state_dict"])

model.eval()

print("\n------inference started, please wait------")

for count, img_name in enumerate(sorted(os.listdir(map_config.img_dir))):
    img_path = os.path.join(map_config.img_dir, img_name)
    pil_img = Image.open(img_path)
    img = map_utils.convert_input_images(pil_img)
    img = torch.unsqueeze(img, 0) # add the batch dim
    output = model(img)
    output = torch.squeeze(output["out"], 0) # remove the batch dim

    output_prediction_grayscale = output.argmax(0)
    output_prediction_rgb = map_utils.put_colors_in_grayscale_mask(output_prediction_grayscale,
                                                            map_config.n_classes, map_config.label_colors)

    output_img = Image.fromarray(np.uint8(output_prediction_rgb), mode="RGB")

    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(pil_img.resize((600, 300), resample=Image.NEAREST))
    ax[0].set_title("Input Image")
    ax[0].set_axis_off()
    ax[1].imshow(output_img)
    ax[1].set_title("Predicted Mask")
    ax[1].set_axis_off()

    op_path = os.path.join(map_config.op_dir, img_name)
    # fig.savefig("predicted_mask.png")
    fig.savefig(op_path)
    plt.close()

    if (count+1)%5 == 0:
        print(f'\n{count+1} images done\n')

print("\n------done------")