import cv2, json

def make_csrt():
    # OpenCV API differs across versions; try common locations
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker not found. Install opencv-contrib-python.")

cap = cv2.VideoCapture("assets/clip.mp4")
with open("assets/init_bbox.json") as f:
    bbox = tuple(json.load(f)["bbox"])  # [x, y, w, h]

ok, frame = cap.read()
tracker = make_csrt()
tracker.init(frame, bbox)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    ok, box = tracker.update(frame)
    if ok:
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("track", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()