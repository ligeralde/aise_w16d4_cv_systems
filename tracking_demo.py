import cv2
import json
import matplotlib.pyplot as plt

def make_csrt():
    # OpenCV API differs across versions
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker not found. Install opencv-contrib-python.")

# -------------------------
# Load video and initial bounding box
# -------------------------
cap = cv2.VideoCapture("assets/clip.mp4")

with open("assets/init_bbox.json") as f:
    bbox = tuple(json.load(f)["bbox"])  # [x, y, w, h]

ok, frame = cap.read()
if not ok:
    raise RuntimeError("Failed to read first frame from video.")

# -------------------------
# Initialize tracker
# -------------------------
tracker = make_csrt()
tracker.init(frame, bbox)

# -------------------------
# Track and visualize
# -------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    ok, box = tracker.update(frame)
    if ok:
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Use OpenCV GUI on Mac (may fail in some environments)
    cv2.imshow("CSRT Tracker", frame)
    
    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
