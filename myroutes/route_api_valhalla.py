import requests

# Function to get Valhalla instructions for the entire route
def get_valhalla_instructions(valhalla_base_url, start_location, end_location):
    # Make a request to Valhalla to get route instructions
    payload = {
        "locations": [
            {"lat": start_location["lat"], "lon": start_location["lon"]},
            {"lat": end_location["lat"], "lon": end_location["lon"]},
        ],
        "costing": "pedestrian",
        "directions_options": {
            "units": "kilometers",
            "language": "en",  # Language for the turn-by-turn directions
            "format": "osrm"
        },
    }

    response = requests.post(f"{valhalla_base_url}/route", json=payload)

    valhalla_instructions = []
    if response.status_code == 200:
        valhalla_data = response.json()

        # Extract locations, instructions and duration for the route
        steps = valhalla_data["routes"][0]["legs"][0]["steps"]
        for step in steps:
            valhalla_instructions.append({
                    "duration": step["duration"],
                    "instruction": step["maneuver"]["instruction"],
                    "location": step["maneuver"]["location"][::-1]
                })
    
    else:
        print(f"Error: {response.status_code} - {response.text}")

    return valhalla_instructions
