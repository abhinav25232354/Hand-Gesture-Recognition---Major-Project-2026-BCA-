import cv2


def apply_visual_effect(image, finger_count: int):
    overlay = image.copy()
    if finger_count == 1:
        color = (255, 220, 160)
        mode = "Precision"
    elif finger_count == 2:
        color = (180, 255, 220)
        mode = "Cut"
    elif finger_count == 3:
        color = (170, 235, 255)
        mode = "Utility"
    elif finger_count >= 4:
        color = (210, 210, 255)
        mode = "Wide"
    else:
        color = (200, 200, 200)
        mode = "Idle"

    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), color, -1)
    image = cv2.addWeighted(overlay, 0.08, image, 0.92, 0)
    return image, mode
