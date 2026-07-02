import cv2
import numpy as np


def apply_pop_art(frame, levels=4):
    multiplier = 255 // (levels - 1)
    quantized = (frame // multiplier) * multiplier

    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 40)  # Push saturation matrix up

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

