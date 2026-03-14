import cv2


def draw_overlay(
    image,
    finger_count: int,
    mode_text: str,
    action_text: str,
    stability_progress: int,
    stability_target: int,
    status_text: str,
):
    cv2.rectangle(image, (0, 0), (540, 105), (15, 15, 15), -1)
    cv2.putText(
        image,
        f"Fingers: {finger_count}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        f"Mode: {mode_text}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        image,
        f"Gesture: {action_text}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        image,
        f"Stability: {stability_progress}/{stability_target} | {status_text}",
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 255, 200),
        1,
    )
