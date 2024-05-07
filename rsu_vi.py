'''
---for video input---
python rsu_vi.py "input/input_video.mp4" "segmentation_output.avi" "model_weights/model.onnx" "input/new.gpx" --headless

---for image folder input---
python rsu_vi.py "images" "segmentation_output" "model_weights/model.onnx" "input/new.gpx"

---for camera input---
python rsu_vi.py 0 "cam_output.avi" "model_weights/model.onnx" "input/new.gpx"
'''

import os
import argparse
import cv2
import numpy as np

from sensation.data_handler import InputHandler, InputType
from sensation.utils.analyze import find_dominant_column
from segmentator import Segmentator

import walk_simulator
from myroutes import map_matching_api_valhalla

def get_instructions(gpx_data_path):
    gpx_data = walk_simulator.load_gpx_data(gpx_data_path)
    coordinates = walk_simulator.extract_coordinates_with_time(gpx_data)
    valhalla_base_url = "https://valhalla1.openstreetmap.de"

    # Get Valhalla instructions for the entire route
    valhalla_instructions = map_matching_api_valhalla.get_valhalla_instructions(valhalla_base_url, coordinates)

    walk_simulator.get_video_instructions_from_coordinates(coordinates, valhalla_instructions)

    return coordinates

def process_images(ih: InputHandler, segmentator: Segmentator, op_folder):
    if not os.path.isdir(op_folder):
        print(f"\nOutput folder i.e. {op_folder} does not exist, please create one\n")
        return
    
    print("\n------processing image folder------\n")
    for count, image_path in enumerate(ih.get_images()):
        image = ih.load_image(image_path)    
        height, width, _ = image.shape
        mask = segmentator.inference(image)
        mask_rgb = segmentator.mask_to_rgb(mask)

        mask_rgb = cv2.resize(mask_rgb, (width, height))

        op_path = os.path.join(op_folder, image_path)

        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        ih.save_image(mask_bgr, op_path)

        if (count+1) % 5 == 0:
            print(f"{count+1} images done")
    print("\n------all images processed------\n")


def start_sensation(ih: InputHandler, segmentator: Segmentator, headless, gps_file_path, start_dominant_column: int):
    print("\n------sensation started------\n")

    cap = ih.get_cap()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_path = ih.get_output_path()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width*3, frame_height))

    # Count frame when to run inference and analyze
    count_frame = 0

    # init mask_rgb to not every time inference and analyze to save resources
    mask_rgb = None
    segmentator_input_frame = None

    # get valhalla instructions
    current_frame = 0
    current_coord_index = 0
    coordinates = get_instructions(gps_file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count_frame == 0 or count_frame > start_dominant_column:
            segmentator_input_frame = frame.copy()
            mask = segmentator.inference(frame) # returns grayscale mask
            mask_rgb = segmentator.mask_to_rgb(mask)
            mask_rgb = cv2.resize(mask_rgb, (frame_width, frame_height))

            # estimate the maximum present of sidewalk pixel
            dominant_column = find_dominant_column(mask_rgb, segmentator.get_sidewalk_rgb())
            # print(f"The dominant column is: {dominant_column}")

            if dominant_column == 1:
                model_command = "Go left"
            elif dominant_column == 2:
                model_command = "Stay center"
            elif dominant_column == 3:
                model_command = "Go right"
            else:
                model_command = "No sidewalk detected in the frame"

            if not headless:
                print(f"The dominant column is: {dominant_column}")
                print(model_command)

            # mask_rgb = visualize_dominant_column(mask_rgb, segmentator.get_sidewalk_rgb(), dominant_column)
            count_frame = 0
        

        elapsed_time = (current_frame // fps)

        if elapsed_time >= coordinates[current_coord_index]['elapsed_time']:
            lat = coordinates[current_coord_index]['lat']
            long = coordinates[current_coord_index]['lon']
            speed = coordinates[current_coord_index]['speed']
            direction_command = coordinates[current_coord_index]['direction_command']
            instruction = coordinates[current_coord_index]['video_instruction']

            current_coord_index += 1
        
        # To check progress
        if headless:
            if (current_frame / fps) % 5 == 0:
                print(f"\n{elapsed_time} seconds of video processed\n")


    # Overlay elapsed time and coordinates on the video frame
        cv2.putText(frame, f"Time: {int(elapsed_time)}s; Speed: {speed:.3f} m/s", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, f"Lat: {lat:.6f}, Lon: {long:.6f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, f"Direction Command: {direction_command}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, f"Valhalla Instruction", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, f"is: {instruction}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, f"Model Command: {model_command}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    

        
        cv2.putText(mask_rgb, f"Output Mask", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(mask_rgb, f"Model command: {model_command}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        
        cv2.putText(segmentator_input_frame, f"Input frame to model", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        op_video_frame = np.hstack((frame, mask_rgb, segmentator_input_frame))
        out.write(op_video_frame)

        current_frame += 1

        count_frame += 1


        if not headless:
            cv2.imshow('Segmented Video', mask_rgb)
            cv2.imshow('Original Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if not headless:
        cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(description='SENSATION system for assistive navigation')
    parser.add_argument('input_path', help='Path to a folder of images, a video file, or a USB camera path')
    parser.add_argument('output_path', help='Output path for segmented images/videos')
    parser.add_argument('model_path', help='Path to the ONNX model')
    parser.add_argument('gps_file_path', help='Path to the GPS coordinate file')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI)')
    parser.add_argument("--motors",
                        action="store_true",
                        default=False,
                        help="Activates vibration motors(Default deactivated)")
    parser.add_argument("--visualize",
                        action="store_true",
                        default=False,
                        help="Visualize dominant column on sidewalk for drift detection.")
    parser.add_argument("--framesteps",
                        type=int,
                        default=30,
                        help="Frame step when to run segmentation on frame(Default: 30)")
    args = parser.parse_args()
    
    print("\n------rsu_vi started------\n")

    ih = InputHandler(input_path=args.input_path,
                      output_path=args.output_path)
    segmentator = Segmentator(model_path=args.model_path,
                              input_height=300,
                              input_width=600)

    input_type = ih.get_input_type()

    if input_type == InputType.IMAGES:
        process_images(ih, segmentator, args.output_path)    
    else:
        start_sensation(ih,
                        segmentator,
                        args.headless,
                        args.gps_file_path,
                        args.framesteps
                        )


if __name__ == "__main__":
    main()
