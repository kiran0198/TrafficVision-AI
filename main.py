from ultralytics import YOLO
import cv2

import math
import torch
from datetime import datetime

#device = "cuda" if torch.cuda.is_available() else "cpu"
# Load YOLO model

model = YOLO("models/yolo26m.pt")
#model.to(device)
#i have a gpu so i will use it to speed up the process if you don't have a gpu you can comment the line below and uncomment the line above
model.to('cuda')

# Open input video
cap = cv2.VideoCapture("videos/input.mp4")

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "output/annotated_output.mp4",
    fourcc,
    fps,
    (width, height)
)

# SPEED FEATURE (Setup variables before the loop) ---
previous_centers = {}
vehicle_speed = {}
ASSUMED_CAR_HEIGHT = 1.6  # Average height of a car in meters

# --- ACCURATE RE-SCALE PER VEHICLE CLASS ---
# Class mapping: 1: Bicycle, 2: Car, 3: Motorcycle, 5: Bus, 7: Truck
class_heights = {0: 1.7, 1: 1.0, 2: 1.5, 3: 1.1, 5: 3.2, 7: 3.5}
# Open a text file to save the logs
log_file = open("output/traffic_log.txt", "w")
log_file.write("Vehicle Speeds\n")
logged_vehicles = set()
interaction_history = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect + Track
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0,1,2,3,5,7],
        conf=0.4
    )

    r = results[0]
    annotated = r.plot()

    objects = []

    if r.boxes.id is not None:

        # Optimization: use .detach().cpu() to prevent massive memory copies
        boxes = r.boxes.xyxy.detach().cpu().numpy()
        ids = r.boxes.id.int().detach().cpu().tolist()
        classes = r.boxes.cls.int().detach().cpu().tolist()

        for box, track_id, cls in zip(boxes, ids, classes):

            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # --- CALCULATE SCALE FOR BOTH SPEED AND RISK ---
            assumed_height = class_heights.get(cls, 1.6) 
            box_height = y2 - y1
            pixels_per_meter = box_height / assumed_height
            # -----------------------------------------------

            # --- NEW: SPEED FEATURE (Calculate and Display Speed) ---
            if track_id in previous_centers:
                px, py = previous_centers[track_id]
                
                # Pixel displacement
                pixel_distance = math.sqrt((cx - px)**2 + (cy - py)**2)
                
                # Avoid division by zero just in case
                if pixels_per_meter > 0:
                    meters = pixel_distance / pixels_per_meter
                    speed_mps = meters * fps
                    speed_kmh = speed_mps * 3.6
                    vehicle_speed[track_id] = speed_kmh

            # Update the center for the next frame
            previous_centers[track_id] = (cx, cy)

            # Display Speed on screen (fetching 0 if no speed is calculated yet)
            speed = vehicle_speed.get(track_id, 0)
            
            # --- IMPROVED TEXT VISIBILITY (Much Bigger & Below ID) ---
            speed_text = f"{speed:.1f} km/h"
            
            # Pushed down a bit more so it doesn't overlap with the larger ID box
            text_pos = (int(x1), int(y1) + 45) 

            # 1. Thick black border (Scale up to 1.5, thickness to 7)
            cv2.putText(annotated, speed_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 10)
            
            # 2. Thin white text inside (Scale up to 1.5, thickness to 3)
            cv2.putText(annotated, speed_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 6)
            # -------------------------------------

            objects.append({
                "id": track_id,
                "class": cls,
                "center": (cx, cy),
                "box": (x1, y1, x2, y2),
                "ppm": pixels_per_meter
            })
    
    #calculate distance 

    risk_pairs = []
    vehicle_max_risk = {obj["id"]: 0 for obj in objects}

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):

            obj1 = objects[i]
            obj2 = objects[j]

            # Ignore person-person pairs
            if obj1["class"] == 0 and obj2["class"] == 0:
                continue

            x1, y1 = obj1["center"]
            x2, y2 = obj2["center"]

            # Distance in pixels
            pixel_distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            # Average the scale of the two vehicles to get a localized perspective
            avg_ppm = (obj1["ppm"] + obj2["ppm"]) / 2
            
            if avg_ppm > 0:
                # Convert pixels to actual meters!
                distance_meters = pixel_distance / avg_ppm
            else:
                distance_meters = 999

            # Interaction zone: Evaluate if they are within 10 meters of each other
            if distance_meters < 10.0:
                risk_pairs.append((obj1, obj2, distance_meters))
    
    #Real-Time Analytics Charts

    interaction_history.append(len(risk_pairs))
    if len(interaction_history) > 100: # Maintain a sliding window of the last 100 frames
        interaction_history.pop(0)
    
    # --- DRAW LIVE ANALYTICS MINICHART (FIXED & CLAMPED) ---
    chart_width = 350
    chart_height = 150 
    graph_origin_x = width - chart_width - 30
    graph_origin_y = height - 30
    chart_top = graph_origin_y - chart_height
    
    # Draw a clean background card
    cv2.rectangle(
        annotated, 
        (graph_origin_x - 20, chart_top - 20), 
        (width - 10, graph_origin_y + 10), 
        (30, 30, 30), 
        -1
    )
    
    # Title and Author
    cv2.putText(annotated, "Live Conflict Waveform", (graph_origin_x, chart_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(annotated, "Author: Kiran K. Sahu", (graph_origin_x, chart_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Plot the history line graph
    x_step = chart_width / 100
    for idx in range(1, len(interaction_history)):
        # Calculate raw Y positions (using * 7 so it fits in the box)
        y1_raw = graph_origin_y - (interaction_history[idx - 1] * 7)
        y2_raw = graph_origin_y - (interaction_history[idx] * 7)
        
        # CLAMP: Ensure Y never goes above the chart top + padding
        y1 = max(chart_top + 40, y1_raw)
        y2 = max(chart_top + 40, y2_raw)
        
        pt1 = (int(graph_origin_x + (idx - 1) * x_step), int(y1))
        pt2 = (int(graph_origin_x + idx * x_step), int(y2))
        
        cv2.line(annotated, pt1, pt2, (255, 100, 0), 2)
    # --------------------------------------------------------

    # Risk Score 
    for obj1, obj2, dist_m in risk_pairs:

        # New Risk Formula: 0 meters = 100 risk, 10 meters = 0 risk
        risk = max(0, 100 - (dist_m * 10))

        # --- Update the max risk for logging ---
        if risk > vehicle_max_risk[obj1["id"]]:
            vehicle_max_risk[obj1["id"]] = risk
        if risk > vehicle_max_risk[obj2["id"]]:
            vehicle_max_risk[obj2["id"]] = risk
        # -------------------------------------------------

        # Color coding based on REAL METERS
        if dist_m <= 2.5:          # Less than 2.5 meters apart (Extreme Danger)
            color = (0,0,255)      # Red
            level = "HIGH"
        elif dist_m <= 5.0:        # Less than 5 meters apart (Tailgating / Close side-by-side)
            color = (0,255,255)    # Yellow
            level = "MEDIUM"
        else:                      # 5 to 10 meters apart (Safe interaction zone)
            color = (0,255,0)      # Green
            level = "LOW"
    
        # Draw Connection Line
        p1 = obj1["center"]
        p2 = obj2["center"]

        cv2.line(
            annotated,
            p1,
            p2,
            color,
            2
        )

        # Show Risk Score
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        
        risk_text = f"{level} ({int(risk)})"
        cv2.putText(annotated, risk_text, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(annotated, risk_text, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # --------------------------------------------------------------
    
    # Save frame data to log file
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- SAVE TO LOG FILE (Only log once per vehicle) ---
    for obj in objects:
        v_id = obj["id"]
        v_speed = vehicle_speed.get(v_id, 0)
        
        # Only log if it has a speed AND hasn't been logged yet
        if v_speed > 0 and v_id not in logged_vehicles:
            log_file.write(f"ID: {v_id}, Speed: {v_speed:.1f} km/h\n")
            logged_vehicles.add(v_id) # Remember: just logged this one :)
    # ----------------------------------------------------

    #Dashboard
    cv2.rectangle(
        annotated,
        (10,10),
        (300,90),
        (40,40,40),
        -1
    )

    cv2.putText(
        annotated,
        f"Vehicles : {len(objects)}",
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        annotated,
        f"Interactions : {len(risk_pairs)}",
        (20,65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    out.write(annotated)

    cv2.imshow("Traffic", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()