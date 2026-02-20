import cv2


def apply_visual_effect(image, finger_count: int):
    mode_text = "Normal"

    if finger_count == 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mode_text = "Greyscale"
    elif finger_count == 2:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h[:] = 140
        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mode_text = "Purple"
    elif finger_count == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h[:] = 10
        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mode_text = "Orange"
    elif finger_count == 4:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h[:] = 25
        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mode_text = "Yellow"

    return image, mode_text

