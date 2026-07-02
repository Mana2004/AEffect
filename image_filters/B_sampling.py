import cv2
import numpy as np


def apply_pixel_art(frame, pixel_size=6):

    h, w = frame.shape[:2]

    low_res = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)

    palette = np.array([
        [15, 15, 27],  # 0. Dark Shadow / Near Black
        [63, 40, 50],  # 1. Deep Shadow Purple
        [156, 75, 89],  # 2. Shadow Mauve
        [214, 142, 133],  # 3. Vibrant Salmon/Warm Midtone
        [244, 210, 194],  # 4. Light Skin Highlight
        [101, 140, 68],  # 5. Retro Green
        [71, 193, 169],  # 6. Vibrant Teal
        [194, 185, 94],  # 7. Dull Yellow
        [242, 230, 184],  # 8. Bright Cream Highlight
        [43, 73, 191],  # 9. Deep Blue
        [89, 141, 240],  # 10. Light Sky Blue
        [36, 153, 219],  # 11. Vibrant Orange
        [45, 103, 181],  # 12. Warm Brown
        [112, 112, 112],  # 13. Neutral Gray
        [189, 189, 189],  # 14. Light Gray
        [245, 245, 245]  # 15. Near White
    ], dtype=np.float32)

    pixels = low_res.reshape((-1, 1, 3)).astype(np.float32)

    pal = palette.reshape((1, -1, 3))

    distances = np.sqrt(np.sum((pixels - pal) ** 2, axis=2))

    closest_color_indices = np.argmin(distances, axis=1)

    quantized_low_res = palette[closest_color_indices].reshape(low_res.shape).astype(np.uint8)

    pixelated_bg = cv2.resize(quantized_low_res, (w, h), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    kernel = np.ones((3, 3), np.uint8)
    edge_gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

    _, binary_edges = cv2.threshold(edge_gradient, 15, 255, cv2.THRESH_BINARY)

    low_res_edges = cv2.resize(binary_edges, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_NEAREST)
    pixelated_edges = cv2.resize(low_res_edges, (w, h), interpolation=cv2.INTER_NEAREST)

    final_art = pixelated_bg.copy()
    final_art[pixelated_edges > 0] = [15, 15, 27]  # Overlay outline using our darkest palette tone

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