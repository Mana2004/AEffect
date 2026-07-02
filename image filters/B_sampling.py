import cv2
import numpy as np

def apply_pixel_art(frame, pixel_size=16):
    h, w = frame.shape[:2]
    low_res = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(low_res, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_halftone(frame, step=8):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255  # Blank white matrix

    for y in range(0, h, step):
        for x in range(0, w, step):
            luminosity = np.mean(gray[y:y + step, x:x + step])
            radius = int((1 - luminosity / 255.0) * (step / 2))
            if radius > 0:
                cv2.circle(canvas, (x + step // 2, y + step // 2), radius, (0, 0, 0), -1)

    return canvas