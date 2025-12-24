import gpxpy
import folium
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radius of the Earth in kilometers

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # in kilometers

    return distance




# Read the GPX file
with open('25-december-2023-anzere.gpx', 'r') as gpx_file:
    gpx = gpxpy.parse(gpx_file)

latitudes = []
longitudes = []
elevations = []
times = []

for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            latitudes.append(point.latitude)
            longitudes.append(point.longitude)
            elevations.append(point.elevation)
            times.append(point.time)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(longitudes, latitudes, label='Snowboarding Path')
# plt.scatter(longitudes, latitudes, c='red', label='Points')  # Points on the path
# plt.title('Snowboarding Runs')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.show()

total_distances = []  # To store total distance of each run
total_times = []  # To store total time of each run

for track in gpx.tracks:
    for segment in track.segments:
        segment_distance = 0
        segment_time = 0
        previous_point = None

        for point in segment.points:
            if previous_point:
                # Calculate distance from the previous point
                segment_distance += haversine(previous_point.latitude, previous_point.longitude, point.latitude, point.longitude)

                # Calculate time difference from the previous point
                if previous_point.time and point.time:
                    time_diff = (point.time - previous_point.time).total_seconds()
                    segment_time += time_diff

            previous_point = point

        total_distances.append(segment_distance)
        total_times.append(segment_time)


average_speeds = []

for i in range(len(total_distances)):
    if total_times[i] > 0:  # To avoid division by zero
        # Convert time from seconds to hours
        hours = total_times[i] / 3600

        # Calculate average speed in km/h
        average_speed = total_distances[i] / hours
        average_speeds.append(average_speed)
    else:
        average_speeds.append(0)


for i, speed in enumerate(average_speeds):
    print(f"Average Speed for Run {i + 1}: {speed:.2f} km/h")




# Calculate the average location to center the map
avg_lat = sum(latitudes) / len(latitudes)
avg_lon = sum(longitudes) / len(longitudes)

# Create a map centered around the average location
map = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)

# Add a line to the map with the snowboarding path
folium.PolyLine(list(zip(latitudes, longitudes)), color="blue", weight=2.5, opacity=1).add_to(map)

map.save("SnowboardingRunsMap.html")


# ... previous code ...

filtered_distances = []  # To store distances of actual runs (not ski lifts)
filtered_times = []  # To store times of actual runs (not ski lifts)
run_indices = []  # To keep track of which runs are actual runs

total_downhill_distance = 0
total_downhill_time = 0

for track_index, track in enumerate(gpx.tracks):
    for segment_index, segment in enumerate(track.segments):
        if segment.points:
            start_point = segment.points[0]
            end_point = segment.points[-1]
            start_elevation = segment.points[0].elevation  # Elevation at the start of the segment
            end_elevation = segment.points[-1].elevation  # Elevation at the end of the segment

            if start_elevation is not None and end_elevation is not None:
                # Check if the run is downhill or flat
                if end_elevation <= start_elevation:
                    start_time = start_point.time
                    segment_distance = 0
                    segment_time = 0
                    previous_point = None

                    for point in segment.points:
                        if previous_point:
                            # Calculate distance from the previous point
                            segment_distance += haversine(previous_point.latitude, previous_point.longitude, point.latitude, point.longitude)

                            # Calculate time difference from the previous point
                            if previous_point.time and point.time:
                                time_diff = (point.time - previous_point.time).total_seconds()
                                segment_time += time_diff

                        previous_point = point

                    filtered_distances.append(segment_distance)
                    filtered_times.append(segment_time)
                    run_indices.append((track_index, segment_index))

                    # Print the run number, date, and time
                    if start_time:
                        print(f"Run {track_index + 1}-{segment_index + 1} started on {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

                total_downhill_distance += segment_distance
                total_downhill_time += segment_time

# ... continue with average speed calculation ...

average_speeds = []
for i in range(len(filtered_distances)):
    if filtered_times[i] > 0:  # To avoid division by zero
        # Convert time from seconds to hours
        hours = filtered_times[i] / 3600

        # Calculate average speed in km/h
        average_speed = filtered_distances[i] / hours
        average_speeds.append(average_speed)
    else:
        average_speeds.append(0)


# Calculate combined average speed if there was any downhill time
if total_downhill_time > 0:
    # Convert total time from seconds to hours
    total_hours = total_downhill_time / 3600

    # Calculate combined average speed in km/h
    combined_average_speed = total_downhill_distance / total_hours
    print(f"Combined Average Speed for all Downhill Runs: {combined_average_speed:.2f} km/h")
else:
    print("No downhill runs detected or no time recorded for downhill runs.")



# Output the results
print("")
print("Average speeds for filtered runds")
for i, speed in enumerate(average_speeds):
    track_index, segment_index = run_indices[i]
    print(f"Average Speed for Run {track_index + 1}-{segment_index + 1}: {speed:.2f} km/h")

