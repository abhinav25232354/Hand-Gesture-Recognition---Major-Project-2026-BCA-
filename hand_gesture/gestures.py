from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


FingerState = Tuple[int, int, int, int, int]


class GestureAction(str, Enum):
    CLOSE_CURRENT_APP = "close_current_app"
    SWITCH_WINDOW = "switch_window"
    CLOSE_ALL_APPS = "close_all_apps"


@dataclass(frozen=True)
class HandInfo:
    finger_state: FingerState
    finger_count: int


def _thumb_open(landmarks, hand_label: Optional[str]) -> int:
    if hand_label == "Right":
        return 1 if landmarks[4].x < landmarks[3].x else 0
    if hand_label == "Left":
        return 1 if landmarks[4].x > landmarks[3].x else 0
    return 1 if landmarks[4].x < landmarks[3].x else 0


def _finger_open(landmarks, tip_id: int) -> int:
    return 1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0


def extract_hand_info(hand_landmarks, hand_label: Optional[str]) -> HandInfo:
    lm = hand_landmarks.landmark
    state: FingerState = (
        _thumb_open(lm, hand_label),
        _finger_open(lm, 8),
        _finger_open(lm, 12),
        _finger_open(lm, 16),
        _finger_open(lm, 20),
    )
    return HandInfo(finger_state=state, finger_count=sum(state))


def map_action(finger_state: FingerState) -> Optional[GestureAction]:
    if finger_state == (0, 0, 0, 0, 0):
        return GestureAction.CLOSE_CURRENT_APP
    if finger_state == (1, 1, 1, 1, 1):
        return GestureAction.SWITCH_WINDOW
    if finger_state == (0, 1, 1, 0, 0):
        return GestureAction.CLOSE_ALL_APPS
    return None


def action_label(action: Optional[GestureAction]) -> str:
    if action == GestureAction.CLOSE_CURRENT_APP:
        return "Close Current App (Fist)"
    if action == GestureAction.SWITCH_WINDOW:
        return "Switch Window (Open Palm)"
    if action == GestureAction.CLOSE_ALL_APPS:
        return "Close All Apps (V Sign)"
    return "None"

