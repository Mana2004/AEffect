import cv2
import numpy as np


def apply_pixel_art(frame, pixel_size=6, total_colors=8):
    h, w = frame.shape[:2]

    low_res = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)

    data = low_res.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(data, total_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    center = center.astype(np.uint8)
    quantized_low_res = center[label.flatten()].reshape(low_res.shape)

    pixelated_bg = cv2.resize(quantized_low_res, (w, h), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    kernel = np.ones((3, 3), np.uint8)
    edge_gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

    _, binary_edges = cv2.threshold(edge_gradient, 20, 255, cv2.THRESH_BINARY)

    low_res_edges = cv2.resize(binary_edges, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_NEAREST)
    pixelated_edges = cv2.resize(low_res_edges, (w, h), interpolation=cv2.INTER_NEAREST)

    final_art = pixelated_bg.copy()
    final_art[pixelated_edges > 0] = [0, 0, 0]

    return final_art


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