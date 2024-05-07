import cv2

def generate_video(input_video_path, output_video_path, coordinates):
    print("\nCreating video(can take upto 10 minutes, 2GB), please wait...\n")
    # Read video file
    video_file = input_video_path
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    width = cap.get(3)
    height = cap.get(4)

    out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (int(width), int(height)))

    # Initialize variables
    current_frame = 0
    current_coord_index = 0

    while True:
        ret, frame_img = cap.read()

        if not ret:
            break  # Break if no more frames

        elapsed_time = (current_frame // fps)

        if elapsed_time >= coordinates[current_coord_index]['elapsed_time']:
            lat = coordinates[current_coord_index]['lat']
            long = coordinates[current_coord_index]['lon']
            speed = coordinates[current_coord_index]['speed']
            direction_command = coordinates[current_coord_index]['direction_command']
            instruction = coordinates[current_coord_index]['video_instruction']

            current_coord_index += 1


    # Overlay elapsed time and coordinates on the video frame
        cv2.putText(frame_img, f"Time: {int(elapsed_time)}s; Speed: {speed:.3f} m/s", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_img, f"Lat: {lat:.6f}, Lon: {long:.6f}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_img, f"Direction Command: {direction_command}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_img, f"Instruction: {instruction}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        # cv2.imshow('Video with GPX Overlay', frame_img)
        out.write(frame_img)

        # Check for 'q' key to exit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        current_frame += 1

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
