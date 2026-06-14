import numpy as np


def interpolate(value, src_min, src_max, dst_min, dst_max):
    """Map a value from one range to another."""
    return np.interp(value, [src_min, src_max], [dst_min, dst_max])


def clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def smooth(current, previous, factor=7):
    """Smooth cursor movement to avoid jitter."""
    smoothX = previous[0] + (current[0] - previous[0]) / factor
    smoothY = previous[1] + (current[1] - previous[1]) / factor
    return smoothX, smoothY


def draw_fps(img, fps):
    """Draw FPS counter on frame."""
    import cv2
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return img


def draw_mode(img, mode_name):
    """Draw current mode label on frame."""
    import cv2
    cv2.putText(img, f'Mode: {mode_name}', (10, 80),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 255), 2)
    return img
