import cv2
import numpy as np


def apply_pop_art(frame, dot_size=4):

    h, w = frame.shape[:2]

    blurred = cv2.bilateralFilter(frame, d=7, sigmaColor=75, sigmaSpace=75)

    bgr_norm = blurred.astype(np.float32) / 255.0
    B, G, R = cv2.split(bgr_norm)

    C = 1.0 - R
    M = 1.0 - G
    Y = 1.0 - B

    K = np.minimum(np.minimum(C, M), Y)

    K_safe = np.clip(K, 0, 0.999)  # Prevent division by zero
    C = (C - K) / (1.0 - K_safe)
    M = (M - K) / (1.0 - K_safe)
    Y = (Y - K) / (1.0 - K_safe)

    K = np.clip((K - 0.2) * 1.5, 0.0, 1.0)

    y_idx, x_idx = np.indices((h, w))

    def halftone_channel(channel, angle_deg, low_t, high_t):

        theta = np.radians(angle_deg)
        scale = np.pi / dot_size

        X = x_idx * np.cos(theta) - y_idx * np.sin(theta)
        Y = x_idx * np.sin(theta) + y_idx * np.cos(theta)

        screen = (np.sin(X * scale) + np.sin(Y * scale) + 2.0) / 4.0

        out = np.zeros_like(channel)

        out[channel > high_t] = 1.0

        dot_mask = (channel > low_t) & (channel <= high_t) & (channel > screen)
        out[dot_mask] = 1.0

        return out

    C_out = halftone_channel(C, 15, low_t=0.2, high_t=0.6)
    M_out = halftone_channel(M, 75, low_t=0.15, high_t=0.6)  # Magenta picks up skin tones
    Y_out = halftone_channel(Y, 0, low_t=0.15, high_t=0.55)  # Yellow triggers easily (for hair/lights)
    K_out = halftone_channel(K, 45, low_t=0.4, high_t=0.8)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 5)
    edges = cv2.bitwise_not(edges)  # Invert so edges are 1s

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    K_out[edges > 0] = 1.0

    final_R = 255.0 * (1.0 - C_out) * (1.0 - K_out)
    final_G = 255.0 * (1.0 - M_out) * (1.0 - K_out)
    final_B = 255.0 * (1.0 - Y_out) * (1.0 - K_out)

    final_art = cv2.merge([final_B, final_G, final_R]).astype(np.uint8)

    return final_art


def apply_posterize(frame, levels=4):
    step = 255.0 / (levels - 1)

    quantized = np.round(frame / step) * step

    return quantized.astype(np.uint8)