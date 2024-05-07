import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import map_config
import map_utils


class Segmentator:
    def __init__(self, model_path, input_height, input_width):
        self.model_path = model_path

    def inference(self, frame):
        frame = Image.fromarray(np.uint8(frame), mode="RGB")
        frame = map_utils.convert_input_images(frame)
        frame = torch.unsqueeze(frame, 0) # add the batch dim

        ort_sess = ort.InferenceSession(self.model_path)
        output = ort_sess.run(None, {'input': frame.numpy()})
        output = np.array(output).squeeze(0).squeeze(0)
        grayscale_mask = output.argmax(0)

        return grayscale_mask

    def mask_to_rgb(self, mask):
        output_prediction_rgb = map_utils.put_colors_in_grayscale_mask(mask, map_config.n_classes,
                                                                       map_config.label_colors)
        output_img = Image.fromarray(np.uint8(output_prediction_rgb), mode="RGB")
        open_cv_image = np.array(output_img)
        
        return open_cv_image

    def get_sidewalk_rgb(self):
        return [244, 35, 232]