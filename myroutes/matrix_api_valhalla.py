import requests

# Function to get Time Distance info between source location(s) and target location(s)
def get_time_distance_matrix(source_coordinates, target_coordinates):
    matrix_api_url = "https://valhalla1.openstreetmap.de/sources_to_targets"
    payload = {
        "sources": source_coordinates,
        "targets": target_coordinates,
        "costing": "pedestrian",
    }

    flag = True
    while flag:
        matrix_api_response = requests.post(matrix_api_url, json=payload)
        try:
            result = matrix_api_response.json()
            flag = False
            return result
        
        except:
            # retry the api call until successfull
            pass