from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


FingerState = Tuple[int, int, int, int, int]


class GestureAction(str, Enum):
    CUT_TARGET_APP = "cut_target_app"
    OPEN_TASK_VIEW = "open_task_view"
    SELECT_TASK_WINDOW = "select_task_window"
    MINIMIZE_TARGET_APP = "minimize_target_app"
    SHOW_DESKTOP = "show_desktop"
    CLOSE_ALL_APPS = "close_all_apps"


@dataclass(frozen=True)
class HandInfo:
    finger_state: FingerState
    finger_count: int
    index_tip: Tuple[float, float]
    palm_center: Tuple[float, float]
    bounding_box_area: float
    palm_scale: float
    hand_label: Optional[str]
    finger_spread: float
    thumb_is_vertical: bool


def _thumb_open(landmarks, hand_label: Optional[str]) -> int:
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    horizontal_open = False
    if hand_label == "Right":
        horizontal_open = thumb_tip.x < thumb_ip.x
    elif hand_label == "Left":
        horizontal_open = thumb_tip.x > thumb_ip.x
    else:
        horizontal_open = abs(thumb_tip.x - thumb_ip.x) > abs(thumb_tip.y - thumb_ip.y)
    stretched = _distance(thumb_tip, index_mcp) > _distance(thumb_mcp, index_mcp) * 0.85
    return 1 if horizontal_open and stretched else 0


def _finger_open(landmarks, tip_id: int) -> int:
    tip = landmarks[tip_id]
    pip = landmarks[tip_id - 2]
    mcp = landmarks[tip_id - 3]
    tip_to_mcp = _distance(tip, mcp)
    pip_to_mcp = _distance(pip, mcp)
    is_extended = tip.y < pip.y < mcp.y
    is_stretched = tip_to_mcp > pip_to_mcp * 1.15
    return 1 if is_extended and is_stretched else 0


def _distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _pair_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def extract_hand_info(hand_landmarks, hand_label: Optional[str]) -> HandInfo:
    lm = hand_landmarks.landmark
    xs = [point.x for point in lm]
    ys = [point.y for point in lm]
    palm_center = (
        (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5.0,
        (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5.0,
    )
    palm_scale = max(_distance(lm[0], lm[9]), _distance(lm[5], lm[17]), 1e-6)
    index_tip = (lm[8].x, lm[8].y)
    middle_tip = (lm[12].x, lm[12].y)
    thumb_is_vertical = lm[4].y < lm[2].y and abs(lm[4].y - lm[2].y) > abs(lm[4].x - lm[2].x)
    state: FingerState = (
        _thumb_open(lm, hand_label),
        _finger_open(lm, 8),
        _finger_open(lm, 12),
        _finger_open(lm, 16),
        _finger_open(lm, 20),
    )
    return HandInfo(
        finger_state=state,
        finger_count=sum(state),
        index_tip=index_tip,
        palm_center=palm_center,
        bounding_box_area=max((max(xs) - min(xs)) * (max(ys) - min(ys)), 0.0),
        palm_scale=palm_scale,
        hand_label=hand_label,
        finger_spread=_pair_distance(index_tip, middle_tip) / palm_scale,
        thumb_is_vertical=thumb_is_vertical,
    )


def map_action(hand_info: HandInfo) -> Optional[GestureAction]:
    finger_state = hand_info.finger_state
    thumb, index, middle, ring, pinky = finger_state

    if index and middle and not ring and not pinky and hand_info.finger_spread >= 0.34:
        return GestureAction.CUT_TARGET_APP
    if not thumb and index and not middle and not ring and not pinky:
        return GestureAction.OPEN_TASK_VIEW
    if finger_state == (0, 0, 0, 0, 0):
        return GestureAction.SELECT_TASK_WINDOW
    if index and middle and ring and not pinky:
        return GestureAction.MINIMIZE_TARGET_APP
    if thumb and not index and not middle and not ring and not pinky and hand_info.thumb_is_vertical:
        return GestureAction.SHOW_DESKTOP
    if not thumb and index and not middle and not ring and pinky:
        return GestureAction.CLOSE_ALL_APPS
    return None


def action_label(action: Optional[GestureAction]) -> str:
    if action == GestureAction.CUT_TARGET_APP:
        return "Cut Target App (Scissors)"
    if action == GestureAction.OPEN_TASK_VIEW:
        return "Open Task View (Point)"
    if action == GestureAction.SELECT_TASK_WINDOW:
        return "Select Window (Fist)"
    if action == GestureAction.MINIMIZE_TARGET_APP:
        return "Minimize Target App (Three Fingers)"
    if action == GestureAction.SHOW_DESKTOP:
        return "Show Desktop (Thumbs Up)"
    if action == GestureAction.CLOSE_ALL_APPS:
        return "Close All Apps (Rock Sign)"
    return "None"
