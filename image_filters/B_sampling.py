import cv2
import numpy as np


def apply_pixel_art(frame, pixel_size=6, total_colors=8):

    h, w = frame.shape[:2]

    low_res = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)

    blurred_low_res = cv2.bilateralFilter(low_res, d=5, sigmaColor=75, sigmaSpace=75)

    low_res_lab = cv2.cvtColor(blurred_low_res, cv2.COLOR_BGR2Lab)
    data = low_res_lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, total_colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    quantized_lab = centers[labels.flatten()].reshape(low_res.shape).astype(np.uint8)

    quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_Lab2BGR)

    pixelated_bg = cv2.resize(quantized_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = np.ones((3, 3), np.uint8)
    edge_gradient = cv2.morphologyEx(blurred_gray, cv2.MORPH_GRADIENT, kernel)

    _, binary_edges = cv2.threshold(edge_gradient, 20, 255, cv2.THRESH_BINARY)

    low_res_edges = cv2.resize(binary_edges, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_NEAREST)
    pixelated_edges = cv2.resize(low_res_edges, (w, h), interpolation=cv2.INTER_NEAREST)

    final_art = pixelated_bg.copy()

    darkest_index = np.argmin(centers[:, 0])
    darkest_bgr = cv2.cvtColor(centers[darkest_index].reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_Lab2BGR)[0, 0]

    final_art[pixelated_edges > 0] = darkest_bgr

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