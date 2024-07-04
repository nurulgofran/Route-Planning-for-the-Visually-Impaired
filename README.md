
# SENSATION: Sidewalk Environment Detection System for Assistive NavigaTION

## Project Overview

The **Road Scene Understanding for the Visually Impaired** initiative aims to enhance the mobility capabilities of blind or visually impaired persons (BVIPs) by ensuring safer and more efficient navigation on pedestrian pathways. Our research team is advancing the development of the **Sidewalk Environment Detection System for Assistive NavigaTION (SENSATION)**.

The primary objective is to refine a specialized apparatus, a chest-mounted bag equipped with an NVIDIA Jetson Nano, which serves as the core computational unit. This device integrates a variety of sensors, including:
- **Tactile Feedback Mechanisms** (vibration motors) for direction indication
- **Optical Sensors** (webcam) for environmental data acquisition
- **Wireless Communication Modules** (Wi-Fi antenna) for internet connectivity
- **Geospatial Positioning Units** (GPS sensors) for real-time location tracking

Despite promising preliminary designs, several technical challenges persist that warrant further investigation.

## Project Goals

We are actively seeking student collaborators to refine the Jetson Nano-fueled SENSATION system. Participants are expected to:
1. Generate navigational pathways in Python based on defined start and endpoint parameters.
2. Implement real-time geospatial tracking to determine immediate coordinates of BVIPs.
3. Conduct optical recording of current coordinates and evaluate sidewalk orientation algorithmically.

The project will be tested from the main train station in Erlangen to the University Library of Erlangen-Nuremberg (Schuhstrasse 1a).

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

#### Contribution Guidelines

```markdown
## Contributing

Thank you for considering contributing to SENSATION! To get started,

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

---
