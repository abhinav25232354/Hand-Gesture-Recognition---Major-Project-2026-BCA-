from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp

from hand_gesture.gestures import HandInfo, extract_hand_info


class VisionEngine:
    def __init__(self, max_num_hands: int, min_detection_confidence: float, min_tracking_confidence: float):
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def process_frame(self, frame) -> tuple:
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self._hands.process(rgb_image)
        rgb_image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        hand_info: Optional[HandInfo] = None
        best_score = float("-inf")
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self._mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style(),
                )
                hand_label = None
                if results.multi_handedness and len(results.multi_handedness) > idx:
                    hand_label = results.multi_handedness[idx].classification[0].label

                candidate = extract_hand_info(hand_landmarks, hand_label)
                center_offset = abs(candidate.palm_center[0] - 0.5) + abs(candidate.palm_center[1] - 0.5)
                score = candidate.bounding_box_area - (center_offset * 0.08)
                if score > best_score:
                    best_score = score
                    hand_info = candidate
        return image, hand_info
