
# SENSATION: Sidewalk Environment Detection System for Assistive NavigaTION

## Project Overview

The **Road Scene Understanding for the Visually Impaired** initiative aims to develop a Sidewalk Environment Detection System for enhancing the mobility capabilities of visually impaired people through the combination of GPS systems and image segmentation techniques refined for sidewalk recognition.

## Methods

- Using Valhalla routing API and GPS tracker signals to determine direction instructions.
- Training DeepLabv3 ResNet50 image segmentation model on Cityscapes dataset (5000 images, 50 cities) in Pytorch.
- Fine-tuning the model on Mapillary dataset (1000 images) to improve sidewalk detection.
- Combining GPS system and image segmentation model to assist navigation.

The project will be tested from the main train station in Erlangen to the University Library of Erlangen-Nuremberg (Schuhstrasse 1a).

## Features

### Core Components

- Sidewalk detection using DeepLabv3+ semantic segmentation
- Real-time navigation instructions using Valhalla routing
- Position estimation and speed calculation
- Intelligent sidewalk guidance system

### System Architecture

#### Input Processing

- Video input support (testing mode)
- Camera input support (live mode)
- GPS coordinate processing
- Command-line interface for flexible input handling

#### Navigation Features

- Three-column sidewalk detection method
- Automated navigation commands
- Real-time speed estimation
- Combined routing and visual guidance

## Technical Requirements

### Environment Setup

To create a sample environment on Windows:

```bash
conda create -n myenv python=3.9
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Running Scripts

1. **Running `rsu_vi.py`:**
   - Ensure `input_video.mp4` and `GPS` file are in the `input` folder.
   - Place `model.onnx` in the `model_weights` folder.

   For video input:
   ```bash
   python rsu_vi.py "input/input_video.mp4" "segmentation_output.avi" "model_weights/model.onnx" "input/new.gpx" --headless
   ```

   For image folder input:
   ```bash
   python rsu_vi.py "images" "segmentation_output" "model_weights/model.onnx" "input/new.gpx"
   ```

   For camera input:
   ```bash
   python rsu_vi.py 0 "cam_output.avi" "model_weights/model.onnx" "input/new.gpx"
   ```

2. **Exporting ONNX Model (`onnx_export.py`):**
   ```bash
   python onnx_export.py --pytorch="fine_tuned_mapillary.ckpt"
   ```

3. **Training Cityscapes Model (`city_training.py`):**
   - Set the `dataset_path` in `city_config.py`
   - Run the script:
   ```bash
   python city_training.py
   ```

4. **Converting Mapillary Masks to Grayscale (`convert_masks_to_grayscale.py`):**
   - Set `json_path`, `masks_path`, and `op_path` in `map_config.py`
   - Run the script:
   ```bash
   python convert_masks_to_grayscale.py
   ```

5. **Fine-tuning on Mapillary Dataset (`training_pipeline.py`):**
   - Set `mapillary_train_path`, `mapillary_val_path`, `mapillary_test_path`, `city_ckpt_path` in `map_config.py`
   - Run the script:
   ```bash
   python training_pipeline.py
   ```

6. **Performing Inference on an Image (`inference.py`):**
   - Set `map_ckp_path`, `img_dir`, and `op_dir` in `map_config.py`
   - Run the script:
   ```bash
   python inference.py
   ```

The following warning can be ignored:
```
[W NNPACK.cpp:64] Could not initialize NNPACK! Reason: Unsupported hardware.
```

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

## Contact

For questions or support, please contact the project team at nurulgofran@gmail.com

## Links to Resources

- [input_video.mp4](https://faubox.rrze.uni-erlangen.de/getlink/fi4SkMw7qgsHNDEmYtSQR5/input_video.mp4)
- [model.onnx](https://faubox.rrze.uni-erlangen.de/getlink/fiQHYEVH7FSYSk9pfskf8o/model.onnx)
- [trained on cityscapes model checkpoint](https://faubox.rrze.uni-erlangen.de/getlink/fiQxx8EmbRenukfSUVyJpY/trained_on_cityscapes.ckpt)
- [fine-tuned on mapillary model checkpoint](https://faubox.rrze.uni-erlangen.de/getlink/fiVwCRYbMxHR2ZnoxcNnXb/fine_tuned_mapillary.ckpt)

## Contributing

Thank you for considering contributing to SENSATION! To get started,

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

---

