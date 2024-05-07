import argparse

import cv2
# import sensation
from sensation.data_handler import InputHandler, InputType
# from sensation.motors.navibelt import NaviBelt
# from sensation.segmentation import Segmentator
# from sensation.utils.analyze import find_dominant_column
# from sensation.utils.visualize import visualize_dominant_column


def process_images(ih: InputHandler, segmentator: Segmentator, visualize: bool = False):
    for image_path in ih.get_images():
        image = ih.load_image(image_path)    
        height, width, _ = image.shape
        mask = segmentator.inference(image)
        mask_rgb = segmentator.mask_to_rgb(mask)

        mask_rgb = cv2.resize(mask_rgb, (width, height))

        if visualize:
            dominant_column = find_dominant_column(mask_rgb, segmentator.get_sidewalk_rgb())
            mask_rgb = visualize_dominant_column(mask_rgb, segmentator.get_sidewalk_rgb(), dominant_column)

        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        ih.save_image(mask_bgr, image_path)


def start_sensation(ih: InputHandler, segmentator: Segmentator, headless, vmotors, start_dominant_column: int):
    cap = ih.get_cap()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_path = ih.get_output_path()
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    # Count frame when to run inference and analyze
    count_frame = 0

    # init mask_rgb to not every time inference and analyze to save resources
    mask_rgb = None

    # Init vibration motors
    vm = None
    if vmotors:
        # add here navibelt
        vm = NaviBelt()

        # Indicate left to right for ready
        vm.go_left()
        vm.go_right()
        vm.stay_center()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count_frame == 0 or count_frame > start_dominant_column:
            mask = segmentator.inference(frame)
            mask_rgb = segmentator.mask_to_rgb(mask)
            mask_rgb = cv2.resize(mask_rgb, (frame_width, frame_height))

            # estimate the maximum present of sidewalk pixel
            dominant_column = find_dominant_column(mask_rgb, segmentator.get_sidewalk_rgb())
            print(f"The dominant column is: {dominant_column}")

            # Send command to specific motor ofnavibelt
            if vmotors:
                if dominant_column == 1:
                    print("Go left")
                    vm.go_left()
                elif dominant_column == 2:
                    print("Stay center")
                    vm.stay_center()
                elif dominant_column == 3:
                    print("Go right")
                    vm.go_right()
                else:
                    print("No sidewalk detected in the frame.")

            mask_rgb = visualize_dominant_column(mask_rgb, segmentator.get_sidewalk_rgb(), dominant_column)
            count_frame = 0

        out.write(mask_rgb)
        count_frame += 1

        if not headless:
            cv2.imshow('Segmented Video', mask_rgb)
            cv2.imshow('Original Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    vm.disconnect()
    if not headless:
        cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(description='SENSATION system for assistive navigation')
    parser.add_argument('input_path', help='Path to a folder of images, a video file, or a USB camera path')
    parser.add_argument('output_path', help='Output path for segmented images/videos')
    parser.add_argument('model_path', help='Path to the ONNX model')
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
    ih = InputHandler(input_path=args.input_path,
                      output_path=args.output_path)
    segmentator = Segmentator(model_path=args.model_path,
                              input_height=256,
                              input_width=512)

    input_type = ih.get_input_type()

    if input_type == InputType.IMAGES:
        process_images(ih, segmentator, args.visualize)    
    else:
        start_sensation(ih,
                        segmentator,
                        args.headless,
                        ih.get_output_path(),
                        args.framesteps
                        )


if __name__ == "__main__":
    main()
