from __future__ import annotations

import time
from typing import Optional

import cv2

from hand_gesture.actions import DesktopActionExecutor
from hand_gesture.config import RuntimeConfig
from hand_gesture.effects import apply_visual_effect
from hand_gesture.gestures import GestureAction, action_label, map_action
from hand_gesture.ui import draw_overlay
from hand_gesture.vision import VisionEngine


class GestureController:
    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.cap = cv2.VideoCapture(self.config.camera_index)
        self.vision = VisionEngine(
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.executor = DesktopActionExecutor(
            close_all_iterations=self.config.close_all_iterations,
            close_all_step_delay_seconds=self.config.close_all_step_delay_seconds,
        )
        self.candidate_action: Optional[GestureAction] = None
        self.consecutive_count = 0
        self.last_action_time = 0.0
        self.status_text = "Ready"

    def _update_stability(self, action: Optional[GestureAction]) -> None:
        if action is None:
            self.candidate_action = None
            self.consecutive_count = 0
            return
        if action == self.candidate_action:
            self.consecutive_count += 1
        else:
            self.candidate_action = action
            self.consecutive_count = 1

    def _try_execute_action(self) -> None:
        if self.candidate_action is None:
            return
        now = time.time()
        cooldown_elapsed = now - self.last_action_time
        if self.consecutive_count < self.config.consecutive_frames_required:
            return
        if cooldown_elapsed < self.config.action_cooldown_seconds:
            self.status_text = f"Cooldown {self.config.action_cooldown_seconds - cooldown_elapsed:.1f}s"
            return

        ok = self.executor.execute(self.candidate_action)
        if ok:
            self.last_action_time = now
            self.status_text = f"Executed: {action_label(self.candidate_action)}"
        else:
            self.status_text = self.executor.last_error or "Action failed"

        self.consecutive_count = 0
        self.candidate_action = None

    def run(self) -> None:
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self._cleanup()
            return

        print("Hand Gesture Recognition started. Press 'q' to quit.")
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                print("Ignoring empty camera frame.")
                continue

            image, hand_info = self.vision.process_frame(frame)
            finger_count = hand_info.finger_count if hand_info else 0
            action = map_action(hand_info.finger_state) if hand_info else None

            self._update_stability(action)
            self._try_execute_action()

            image, mode_text = apply_visual_effect(image, finger_count)
            draw_overlay(
                image=image,
                finger_count=finger_count,
                mode_text=mode_text,
                action_text=action_label(action),
                stability_progress=self.consecutive_count,
                stability_target=self.config.consecutive_frames_required,
                status_text=self.status_text,
            )

            cv2.imshow("Hand Gesture Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._cleanup()

    def _cleanup(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.vision.close()
        cv2.destroyAllWindows()

