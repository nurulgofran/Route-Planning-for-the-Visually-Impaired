import time
from xml.dom import minidom
from haversine import haversine, Unit
from myroutes import matrix_api_valhalla

# Function to load GPS data from a GPX file
def load_gpx_data(file_path):
    with open(file_path, 'r') as file:
        data = minidom.parse(file)
    return data

# Function to extract GPS coordinates and time from GPX data
def extract_coordinates_with_time(gpx_data):
    track_points = gpx_data.getElementsByTagName('trkpt')
    coordinates = []

    start_time = None
    prev_coords = None
    for idx, point in enumerate(track_points):
        lat = float(point.getAttribute('lat'))
        lon = float(point.getAttribute('lon'))
        time_str = point.getElementsByTagName('time')[0].firstChild.nodeValue.strip()

        current_time = time.mktime(time.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ"))

        if start_time is None:
            start_time = current_time

        elapsed_time = current_time - start_time

        if prev_coords is not None:
            coords1 = (prev_coords['lat'], prev_coords['lon'])
            coords2 = (lat, lon)
            distance = haversine(coords1, coords2, unit= Unit.METERS)
            time_diff = elapsed_time - prev_coords['elapsed_time']

            if distance > 1:
                speed = distance / time_diff
            else:
                speed = 0
        else:
            speed = 0

        coordinates.append({'lat': lat, 'lon': lon, 'elapsed_time': elapsed_time, 'speed': speed})
        prev_coords = {'lat': lat, 'lon': lon, 'elapsed_time': elapsed_time}

    return coordinates

def get_video_instructions_from_coordinates(coordinates, valhalla_instructions):
    instruction_count = 0
    flag = True
    for coordinate in coordinates:
        if instruction_count == len(valhalla_instructions) - 1:
            flag = False
        
        current_gps_coordinate = (coordinate["lat"], coordinate["lon"])
        instruction_info = valhalla_instructions[instruction_count]
        action_coordinate = (instruction_info["location"][0], instruction_info["location"][1])
        
        distance = haversine(current_gps_coordinate, action_coordinate, unit= Unit.METERS)

        # coordinate["haver_dist"] = distance

        if (distance <= 11) and (flag == True): #meters
            instruction_str = instruction_info["instruction"]
            coordinate["video_instruction"] = instruction_str

            if "left" in instruction_str.lower():
                coordinate["direction_command"] = "Go left"

            elif "right" in instruction_str.lower():
                coordinate["direction_command"] = "Go right"

            else:
                coordinate["direction_command"] = "Stay center"

            instruction_count += 1

        else:
            coordinate["video_instruction"] = instruction_str
            coordinate["direction_command"] = "Stay center"

    coordinates[-1]["video_instruction"] = instruction_info["instruction"]

# Function to simulate walking using GPS coordinates and print instructions from Valhalla
def simulate(coordinates, valhalla_instructions):
    print("\nWalking simulator started\n")

    instruction_count = 0
    flag = True
    for coordinate in coordinates:
        if instruction_count == len(valhalla_instructions) - 1:
            flag = False

        # get source location latitude and longitude
        source_coordinate = [{"lat": coordinate["lat"], "lon": coordinate["lon"]}]

        # get target location latitude and longitude
        instruction_info = valhalla_instructions[instruction_count]
        target_coordinate = [{"lat": instruction_info["location"][0],
                            "lon": instruction_info["location"][1]}]
        
        # make the api call to get time-distance info from Valhalla
        result = matrix_api_valhalla.get_time_distance_matrix(source_coordinate, target_coordinate)

        #compare distance anf print appropriate instruction
        if result["sources_to_targets"][0][0]["distance"] <= 0.005 and flag == True:
            instruction_str = instruction_info["instruction"]

            if "left" in instruction_str.lower():
                print(f'Go left\t Current position:{coordinate}\n')

            elif "right" in instruction_str.lower():
                print(f'Go right\t Current position:{coordinate}\n')

            else:
                print(f'Stay center\t Current position:{coordinate}\n')
            
            print(f'********{instruction_info["instruction"]}********\n')
            instruction_count += 1
        
        else:
            print(f'Stay center\t Current position:{coordinate}\n')

    print(f'********{instruction_info["instruction"]}********\n')
