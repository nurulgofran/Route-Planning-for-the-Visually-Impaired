import requests
import matplotlib.pyplot as plt

url = "https://valhalla1.openstreetmap.de/height"
    
# Input data
payload = {
    "range": True,
    "shape": [
        {"lat": 40.712431, "lon": -76.504916},
        {"lat": 40.712275, "lon": -76.605259},
        {"lat": 40.712122, "lon": -76.805694},
        {"lat": 40.722431, "lon": -76.884916},
        {"lat": 40.812275, "lon": -76.905259},
        {"lat": 40.912122, "lon": -76.965694}
    ],
}

# Send GET request to the Valhalla API
response = requests.get(url, json = payload)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    elevation_data = response.json()

    # Extract range and visualize the elevation data graphically
    range_height = elevation_data.get('range_height', [])

    # Display range and elevation data
    print("Elevation Data:")
    for i, (range_val, point) in enumerate(zip(range_height, payload["shape"])):
        elevation = range_val[1]
        cumulative_range_meters = range_val[0]
        cumulative_range_km = cumulative_range_meters / 1000  # Convert meters to kilometers
        print(f"Point {i + 1}:")
        print(f"  Latitude: {point['lat']}")
        print(f"  Longitude: {point['lon']}")
        print(f"  Elevation: {elevation} meters")
        print(f"  Cumulative Range: {cumulative_range_km:.2f} kilometers")
        print("---")

    # Visualize the elevation data graphically
    ranges_km, elevations = zip(*[(r/1000, e) for r, e in range_height])
    plt.plot(ranges_km, elevations, marker='o')
    
    # Label each point in the graph
    for i, (range_km, elevation) in enumerate(zip(ranges_km, elevations)):
        plt.annotate(f'Point {i + 1}\n({range_km:.2f} km, {elevation} m)',
                     (range_km, elevation),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.title('Elevation Profile')
    plt.xlabel('Cumulative Range (kilometers)')
    plt.ylabel('Elevation (meters)')
    plt.show()

else:
    print(f"Error: Unable to retrieve elevation data. Status code: {response.status_code}")
