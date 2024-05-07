import requests

# Function to get Valhalla instructions for the entire route
def get_valhalla_instructions(valhalla_base_url, coordinates):
    # Make a request to Valhalla to get route instructions
    payload = {
        "shape": coordinates,
        "costing": "pedestrian",
        "shape_match": "walk_or_snap",
        "trace_options": {
            "turn_penalty_factor":500
        },
        "directions_options": {
            "units": "kilometers",
            "language": "en",
            "format": "osrm"
        }
    }

    response = requests.post(f"{valhalla_base_url}/trace_route", json=payload)

    valhalla_instructions = []
    if response.status_code == 200:
        valhalla_data = response.json()

        # Extract locations, instructions and duration for the route
        steps = valhalla_data["matchings"][0]["legs"][0]["steps"]
        for step in steps:
            valhalla_instructions.append({
                    "duration": step["duration"],
                    "instruction": step["maneuver"]["instruction"],
                    "location": step["maneuver"]["location"][::-1]
                })
    
    else:
        print(f"Error: {response.status_code} - {response.text}")

    return valhalla_instructions