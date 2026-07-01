import cv2
import numpy as np
import os

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def overlay_transparent(background, overlay, x, y, overlay_width, overlay_height):
    overlay_resized = cv2.resize(overlay, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)

    h,w = background.shape[:2]
    if x>= w or y >= h or x + overlay_width <= 0 or y + overlay_height <= 0:
        return background

    x1, x2 = max(0,x), min(w, x + overlay_width)
    y1, y2 = max(0,y), min(h, y + overlay_height)

    ox1, ox2 = max(0, -x), overlay_width - max(0, (x + overlay_width) - w)
    oy1, oy2 = max(0, -y), overlay_height - max(0, (y + overlay_height) - h)

    sub_bg = background[y1:y2, x1:x2]
    sub_overlay = overlay[oy1:oy2, ox1:ox2]

    alpha = sub_overlay[:, :, 3] / 255.0
    alpha_inverse = 1.0 - alpha

    for i in range(3)
        background[y1:y2, x1:x2, i] = (alpha * sub_overlay[:, :, i] + alpha_inverse * sub_overlay[:, :, i])

    return background

def apply_face_accessory(frame, overlay_img_path, asset_type = "glasses"):
    if not os.path.exists(overlay_img_path):
        return frame

    overlay = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

    for (x,y,w,h) in faces:
        if asset_type == "glasses":
            overlay_w = int(w * 1.0)
            overlay_h = int(h * 0.35)
            ox = x
            oy = y + int (h * 0.25)
        else:
            continue

        frame = overlay_transparent(overlay, gray, ox, oy, overlay_w, overlay_h)

    return frame