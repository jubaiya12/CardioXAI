import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

def digitize_ecg_image(image_path, lead_row=0):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Crop header out
    top_crop = int(h * 0.22)
    bottom_crop = int(h * 0.93)
    ecg_area = gray[top_crop:bottom_crop, :]
    ecg_h, ecg_w = ecg_area.shape

    # 8 lead rows
    row_height = ecg_h // 8
    row_start = lead_row * row_height
    row_end = row_start + row_height
    lead_strip = ecg_area[row_start:row_end, :]
    strip_h, strip_w = lead_strip.shape

    # Threshold to get only dark signal line
    _, binary = cv2.threshold(lead_strip, 80, 255, cv2.THRESH_BINARY_INV)

    # Remove horizontal grid lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    grid_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    binary_clean = cv2.subtract(binary, grid_lines)

    # For each column find the signal line position
    signal = []
    for col in range(strip_w):
        col_pixels = binary_clean[:, col]
        dark_rows = np.where(col_pixels > 0)[0]
        if len(dark_rows) > 0:
            y_pos = np.mean(dark_rows)
        else:
            y_pos = strip_h / 2
        signal.append(y_pos)

    signal = np.array(signal, dtype=float)

    # Invert — higher y = lower voltage
    signal = -signal

    # Auto-correct inversion
    if np.mean(signal[signal > np.percentile(signal, 75)]) < 0:
        signal = -signal

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # Smooth
    window = min(51, len(signal) // 10)
    if window % 2 == 0:
        window += 1
    signal = savgol_filter(signal, window_length=window, polyorder=3)

    return signal

def segment_beats_from_signal(signal, samples_per_beat=180):
    min_distance = max(30, len(signal) // 20)
    peaks, props = find_peaks(
        signal,
        distance=min_distance,
        prominence=0.5,
        height=np.mean(signal) + 0.3 * np.std(signal)
    )

    beats = []
    for peak in peaks:
        start = peak - samples_per_beat // 2
        end = peak + samples_per_beat // 2
        if start >= 0 and end <= len(signal):
            beat = signal[start:end]
            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
            beats.append(beat)

    return np.array(beats) if len(beats) > 0 else np.array([]), peaks

def process_ecg_image(image_path, lead_row=0):
    signal = digitize_ecg_image(image_path, lead_row=lead_row)
    beats, peaks = segment_beats_from_signal(signal)
    return signal, beats, peaks