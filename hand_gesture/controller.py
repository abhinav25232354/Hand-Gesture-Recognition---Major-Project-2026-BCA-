from __future__ import annotations

import logging
import time
from typing import Optional

import cv2

from hand_gesture.actions import DesktopActionExecutor
from hand_gesture.config import RuntimeConfig
from hand_gesture.effects import apply_visual_effect
from hand_gesture.gestures import GestureAction, action_label, map_action
from hand_gesture.ui import draw_overlay
from hand_gesture.vision import VisionEngine

logger = logging.getLogger(__name__)


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
        self.last_index_tip: Optional[tuple[float, float]] = None
        self.last_switch_nav_time = 0.0
        self.switch_motion_accum = (0.0, 0.0)
        self.frame_index = 0

    def _update_stability(self, action: Optional[GestureAction]) -> None:
        if action is None:
            if self.candidate_action is not None:
                logger.debug("Stability reset: previous_candidate=%s", self.candidate_action.value)
            self.candidate_action = None
            self.consecutive_count = 0
            return
        if action == self.candidate_action:
            self.consecutive_count += 1
            logger.debug(
                "Stability tick: action=%s consecutive=%d/%d",
                action.value,
                self.consecutive_count,
                self.config.consecutive_frames_required,
            )
        else:
            self.candidate_action = action
            self.consecutive_count = 1
            logger.debug("New stability candidate: action=%s", action.value)

    def _try_execute_action(self) -> None:
        if self.candidate_action is None:
            return
        now = time.time()
        cooldown_elapsed = now - self.last_action_time
        if self.consecutive_count < self.config.consecutive_frames_required:
            return
        cooldown_required = self.config.action_cooldown_seconds
        if (
            self.candidate_action == GestureAction.SELECT_TASK_WINDOW
            and self.executor.task_view_active
        ):
            cooldown_required = 0.0

        if cooldown_elapsed < cooldown_required:
            self.status_text = f"Cooldown {cooldown_required - cooldown_elapsed:.1f}s"
            logger.debug(
                "Execution blocked by cooldown: action=%s remaining=%.2fs",
                self.candidate_action.value,
                cooldown_required - cooldown_elapsed,
            )
            return

        logger.info("Executing action: %s", self.candidate_action.value)
        ok = self.executor.execute(self.candidate_action)
        if ok:
            self.last_action_time = now
            self.status_text = f"Executed: {action_label(self.candidate_action)}"
            logger.info("Action executed successfully: %s", self.candidate_action.value)
        else:
            self.status_text = self.executor.last_error or "Action failed"
            logger.error("Action failed: %s", self.status_text)

        self.consecutive_count = 0
        self.candidate_action = None

    def _handle_task_view_navigation(self, hand_info) -> None:
        if not self.executor.task_view_active:
            self.last_index_tip = None
            self.switch_motion_accum = (0.0, 0.0)
            return

        if hand_info is None:
            self.last_index_tip = None
            self.switch_motion_accum = (0.0, 0.0)
            return

        if hand_info.finger_state != (0, 1, 0, 0, 0):
            self.last_index_tip = None
            self.switch_motion_accum = (0.0, 0.0)
            return

        current_tip = hand_info.index_tip
        if self.last_index_tip is None:
            self.last_index_tip = current_tip
            return

        # Freeze navigation while fist selection is being stabilized.
        if self.candidate_action == GestureAction.SELECT_TASK_WINDOW:
            self.last_index_tip = current_tip
            return

        frame_dx = current_tip[0] - self.last_index_tip[0]
        frame_dy = current_tip[1] - self.last_index_tip[1]
        if abs(frame_dx) < self.config.switch_nav_frame_deadzone:
            frame_dx = 0.0
        if abs(frame_dy) < self.config.switch_nav_frame_deadzone:
            frame_dy = 0.0
        accum_dx = self.switch_motion_accum[0] + frame_dx
        accum_dy = self.switch_motion_accum[1] + frame_dy
        self.switch_motion_accum = (accum_dx, accum_dy)
        logger.debug(
            "TaskView motion: frame_dx=%.4f frame_dy=%.4f accum_dx=%.4f accum_dy=%.4f",
            frame_dx,
            frame_dy,
            accum_dx,
            accum_dy,
        )

        now = time.time()
        if now - self.last_switch_nav_time < self.config.switch_nav_cooldown_seconds:
            self.last_index_tip = current_tip
            return

        threshold = self.config.switch_nav_min_delta
        direction = None

        if abs(accum_dx) >= abs(accum_dy) and abs(accum_dx) >= threshold:
            direction = "right" if accum_dx > 0 else "left"
        elif abs(accum_dy) > abs(accum_dx) and abs(accum_dy) >= threshold:
            direction = "down" if accum_dy > 0 else "up"

        if direction and self.executor.navigate_task_view(direction):
            self.last_switch_nav_time = now
            self.status_text = f"Task View move: {direction}"
            self.switch_motion_accum = (0.0, 0.0)
            logger.info("Task View navigation: direction=%s", direction)

        self.last_index_tip = current_tip

    def run(self) -> None:
        if not self.cap.isOpened():
            logger.error("Could not open webcam.")
            self._cleanup()
            return

        logger.info("Hand Gesture Recognition started. Press 'q' to quit.")
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                logger.warning("Ignoring empty camera frame.")
                continue

            self.frame_index += 1
            self.executor.refresh_external_target()
            image, hand_info = self.vision.process_frame(frame)
            finger_count = hand_info.finger_count if hand_info else 0
            action = map_action(hand_info.finger_state) if hand_info else None
            logger.debug(
                "Frame %d: finger_count=%d finger_state=%s mapped_action=%s task_view_active=%s",
                self.frame_index,
                finger_count,
                hand_info.finger_state if hand_info else None,
                action.value if action else None,
                self.executor.task_view_active,
            )

            self._update_stability(action)
            self._try_execute_action()
            self._handle_task_view_navigation(hand_info)

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
                logger.info("Quit requested via keyboard.")
                break

        self._cleanup()

    def _cleanup(self) -> None:
        logger.info("Cleaning up camera, vision engine, and UI windows.")
        if self.cap is not None:
            self.cap.release()
        self.vision.close()
        cv2.destroyAllWindows()
