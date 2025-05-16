import cv2
import numpy as np
import torch
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from collections import deque
import time
import os

PED_MODEL_WEIGHTS = "yolov10s.pt"
TL_MODEL_WEIGHTS  = "best.pt"
VIDEO_IN      = "night_example.mp4"
VIDEO_OUT     = "output.mp4"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE   = (640, 384)
TARGET_FPS    = 5

crosswalk_poly = Polygon([(4, 854), (494, 762), (784, 902), (19, 1078)])
crosswalk_poly_right = Polygon([(1055, 903), (1456, 798), (1918, 911), (1902, 1071), (1437, 1077)])
road_poly      = Polygon([(0, 490), (1919, 490), (1919, 1080), (0, 1080)])
tl_zone_poly   = Polygon([(54, 384), (309, 386), (324, 580), (50, 604)])
tl_zone_poly_right   = Polygon([(1217, 336), (1220, 518), (1503, 516), (1494, 322)])

CONF_THRES  = 0.35
IOU_THRES   = 0.4

IMG_DIR          = "jaywalk_images"
os.makedirs(IMG_DIR, exist_ok=True)

GREEN_LO, GREEN_HI = (50, 50, 120), (90, 255, 255)

MIN_RATIO = 0.23
SMOOTH_WIN = 5         # median filter smoothing
MIN_GREEN_PIXELS = 200
MIN_GREEN_ARROW_PIXELS = 40
MIN_GREEN_PIXELS_RIGHT = 40 
MIN_GREEN_ARROW_PIXELS_RIGHT = 10

ped_model = YOLO(PED_MODEL_WEIGHTS).to(DEVICE)
ped_model.fuse()

tl_model  = YOLO(TL_MODEL_WEIGHTS).to(DEVICE)
tl_model.fuse()

cap       = cv2.VideoCapture(VIDEO_IN)
orig_fps  = cap.get(cv2.CAP_PROP_FPS)
sample_interval = int(round(orig_fps / TARGET_FPS))
width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = None
if VIDEO_OUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, TARGET_FPS, (width, height))

light_history = deque(maxlen=SMOOTH_WIN)
light_history_right = deque(maxlen=SMOOTH_WIN)
track_flagged = {}
jaywalk_log   = []

def get_light_state_from_box(orig_frame, box, sx, sy, location):
    x1_r, y1_r, x2_r, y2_r = map(int, box)
    x1 = int(x1_r * sx); y1 = int(y1_r * sy)
    x2 = int(x2_r * sx); y2 = int(y2_r * sy)

    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN", float("inf")
    roi = orig_frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    h, w = mask.shape
    mid_x = w // 2 if location == 'left' else w // 3
    mid_y = int(h * 2/3)
    bl = mask[mid_y:h, 0:mid_x]
    br = mask[mid_y:h, mid_x:w]
    bl_cnt = cv2.countNonZero(bl)
    br_cnt = cv2.countNonZero(br)
    min_green_pixels = MIN_GREEN_PIXELS if location == 'left' else MIN_GREEN_PIXELS_RIGHT
    min_green_arrow_pixels = MIN_GREEN_ARROW_PIXELS if location == 'left' else MIN_GREEN_ARROW_PIXELS_RIGHT

    ratio = float("inf") if br_cnt == 0 else bl_cnt / br_cnt

    NIGHT_PIXELS = 300 if location == 'left' else 100
    MIN_RATIO = 0.42 if bl_cnt >= NIGHT_PIXELS else 0.23

    if bl_cnt >= min_green_arrow_pixels and br_cnt >= min_green_pixels and MIN_RATIO <= ratio <= 2.00:
        return "GREEN_PROTECTED_LEFT", ratio
    elif bl_cnt >= min_green_arrow_pixels and ratio >= 2.00:
        return "PROTECTED_LEFT", ratio
    elif br_cnt >= min_green_pixels:
        return "GREEN", ratio
    return "UNKNOWN", ratio

frame_idx  = 0
start_time = time.time()
event_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % sample_interval != 0:
        frame_idx += 1
        continue

    resized = cv2.resize(frame, TARGET_SIZE)
    sx, sy  = frame.shape[1]/TARGET_SIZE[0], frame.shape[0]/TARGET_SIZE[1]

    scaled_zone = Polygon([(int(x/sx), int(y/sy)) for x,y in tl_zone_poly.exterior.coords])
    scaled_zone_right = Polygon([(int(x/sx), int(y/sy)) for x,y in tl_zone_poly_right.exterior.coords])

    tl_box = None
    tl_res = tl_model(resized,
                      conf=CONF_THRES, iou=IOU_THRES,
                      classes=[0], device=DEVICE)[0]
    if tl_res.boxes.xyxy is not None and len(tl_res.boxes.xyxy) > 0:
        coords = tl_res.boxes.xyxy.cpu().numpy().astype(int)
        for bx1,by1,bx2,by2 in coords:
            cx, cy = (bx1 + bx2)//2, (by1 + by2)//2
            if scaled_zone.contains(Point(cx, cy)):
                tl_box = (bx1, by1, bx2, by2)
                break

    if tl_res.boxes.xyxy is not None and len(tl_res.boxes.xyxy) > 0:
        coords = tl_res.boxes.xyxy.cpu().numpy().astype(int)
        for bx1,by1,bx2,by2 in coords:
            cx, cy = (bx1 + bx2)//2, (by1 + by2)//2
            if scaled_zone_right.contains(Point(cx, cy)):
                tl_box_right = (bx1, by1, bx2, by2)
                break

    if tl_box:
        light_now, ratio = get_light_state_from_box(frame, tl_box, sx, sy, 'left')
        ox1 = int(tl_box[0] * sx); oy1 = int(tl_box[1] * sy)
        ox2 = int(tl_box[2] * sx); oy2 = int(tl_box[3] * sy)
        cv2.rectangle(frame, (ox1,oy1), (ox2,oy2), (0,128,255), 2)
        cv2.putText(frame, light_now, (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        ratio_txt = f"BL/BR green ratio: {ratio:.2f}"
        cv2.putText(frame, ratio_txt, (ox1, oy1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2, cv2.LINE_AA)
    else:
        light_now = "UNKNOWN"

    if tl_box_right:
        light_now_right, ratio = get_light_state_from_box(frame, tl_box_right, sx, sy, 'right')
        ox1 = int(tl_box_right[0] * sx); oy1 = int(tl_box_right[1] * sy)
        ox2 = int(tl_box_right[2] * sx); oy2 = int(tl_box_right[3] * sy)
        cv2.rectangle(frame, (ox1,oy1), (ox2,oy2), (0,128,255), 2)
        cv2.putText(frame, light_now_right, (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        ratio_txt = f"BL/BR green ratio: {ratio:.2f}"
        cv2.putText(frame, ratio_txt, (ox1, oy1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2, cv2.LINE_AA)
    else:
        light_now_right = "UNKNOWN"

    light_history.append(light_now)
    light_state = sorted(light_history)[len(light_history)//2]
    disp_color = {"GREEN_PROTECTED_LEFT":(0,255,0), "GREEN":(0,255,0), "PROTECTED_LEFT":(0,255,255), "UNKNOWN":(0,0,255)}[light_state]
    cv2.putText(frame, f'Light: {light_state}', (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, disp_color, 3)

    light_history_right.append(light_now_right)
    light_state_right = sorted(light_history_right)[len(light_history_right)//2]
    disp_color = {"GREEN_PROTECTED_LEFT":(0,255,0), "GREEN":(0,255,0), "PROTECTED_LEFT":(0,255,255), "UNKNOWN":(0,0,255)}[light_state_right]
    cv2.putText(frame, f'Right Light: {light_state_right}', (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, disp_color, 3)
    
    ped = ped_model.track(resized,
                          conf=CONF_THRES, iou=IOU_THRES,
                          persist=True, classes=[0], device=DEVICE)[0]
    if ped.boxes.id is not None:
        ids   = ped.boxes.id.cpu().numpy().astype(int)
        boxes = ped.boxes.xyxy.cpu().numpy().astype(int)
        for tid, box in zip(ids, boxes):
            x1 = int(box[0] * sx); y1 = int(box[1] * sy)
            x2 = int(box[2] * sx); y2 = int(box[3] * sy)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            inside = crosswalk_poly.contains(Point(cx,cy))
            if inside and (light_state == "GREEN" or light_state == "GREEN_PROTECTED_LEFT" or light_state == "PROTECTED_LEFT" or light_state_right == "GREEN_PROTECTED_LEFT" or light_state_right == "PROTECTED_LEFT") and not track_flagged.get(tid, False):
                # record event image
                event_time = frame_idx / orig_fps
                tag = f"event_{event_counter}_{event_time:.2f}s"
                img_path = os.path.join(IMG_DIR, f"{tag}.jpg")
                cv2.imwrite(img_path, frame)
                jaywalk_log.append({
                    'id': tid,
                    'frame': frame_idx,
                    'time': event_time,
                    'image': img_path
                })
                event_counter += 1
                track_flagged[tid] = True
            inside_right = crosswalk_poly_right.contains(Point(cx,cy))
            if inside_right and (light_state_right == "GREEN" or light_state_right == "GREEN_PROTECTED_LEFT" or light_state_right == "PROTECTED_LEFT" or light_state == "PROTECTED_LEFT") and not track_flagged.get(tid, False):
                event_time = frame_idx / orig_fps
                tag = f"event_{event_counter}_{event_time:.2f}s"
                img_path = os.path.join(IMG_DIR, f"{tag}.jpg")
                cv2.imwrite(img_path, frame)
                jaywalk_log.append({
                    'id': tid,
                    'frame': frame_idx,
                    'time': event_time,
                    'image': img_path
                })
                event_counter += 1
                track_flagged[tid] = True
            col = (0,0,255) if (inside or inside_right) else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, f'ID {tid}', (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.circle(frame, (cx, cy), 3, col, -1)

    cv2.polylines(frame, [np.array(crosswalk_poly.exterior.coords, int)], True, (255,255,0), 2)
    cv2.polylines(frame, [np.array(crosswalk_poly_right.exterior.coords, int)], True, (255,255,0), 2)
    cv2.polylines(frame, [np.array(road_poly.exterior.coords, int)],      True, (200,200,200), 1)
    cv2.polylines(frame, [np.array(tl_zone_poly.exterior.coords, int)],   True, (0,128,255),   2)
    cv2.polylines(frame, [np.array(tl_zone_poly_right.exterior.coords, int)],   True, (0,128,255),   2)

    if writer:
        writer.write(frame)
    cv2.imshow("Jaywalk & Light Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

print(f"\nDONE in {time.time()-start_time:.1f}s")
print(f"Detected {len(jaywalk_log)} jaywalking events:")
for evt in jaywalk_log:
    print(f" • Track {evt['id']} at {evt['frame']} frames ({evt['time']:.2f}s), image={evt['image']}")
