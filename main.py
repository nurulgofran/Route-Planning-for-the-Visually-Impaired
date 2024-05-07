import time
from pprint import pprint
import walk_simulator
from myroutes import map_matching_api_valhalla
import generate_video_walk

gpx_data_path = 'new.gpx' # Path to gps data file
gpx_data = walk_simulator.load_gpx_data(gpx_data_path)
coordinates = walk_simulator.extract_coordinates_with_time(gpx_data)

valhalla_base_url = "https://valhalla1.openstreetmap.de"
start_location = {"lat": 49.59613, "lon": 11.00242}  # Erlangen HBF
end_location = {"lat": 49.59693, "lon": 11.00711}  # Lib

# Get Valhalla instructions for the entire route
valhalla_instructions = map_matching_api_valhalla.get_valhalla_instructions(valhalla_base_url, coordinates)
print("\nRoute info from valhalla\n")
pprint(valhalla_instructions)


walk_simulator.get_video_instructions_from_coordinates(coordinates, valhalla_instructions)
    

input_video_path = "input_video.mp4"
output_video_path = "output_video.avi"
generate_video_walk.generate_video(input_video_path, output_video_path, coordinates)
print("\nVideo successfully created as output_video.avi in current directory\n")


# print("\nStarting walk simulator...\n")
# time.sleep(5)
# # Start the walk simulation
# walk_simulator.simulate(coordinates, valhalla_instructions)
