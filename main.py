from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import pyautogui


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0


Point = Tuple[float, float]
FingerState = Tuple[int, int, int, int, int]


@dataclass
class FrameSample:
    timestamp: float
    palm_center: Point
    index_tip: Point
    middle_tip: Point
    hand_size_px: float
    finger_state: FingerState
    gesture_flags: Dict[str, bool]
    velocity: Point
    velocity_mag: float
    horizontal_velocity: float
    vertical_velocity: float
    pinch_distance: float
    v_spread: float
    confidence: float


@dataclass
class GestureFSM:
    entered_at: float = 0.0
    active: bool = False
    meta: Dict[str, float] = field(default_factory=dict)


class GestureController:
    def __init__(self) -> None:
        self.frame_width = 640
        self.frame_height = 480
        self.target_fps = 30
        self.min_confidence = 0.7
        self.landmark_average_window = 5
        self.action_vote_window = 7
        self.screen_w, self.screen_h = pyautogui.size()
        self.overlay_lines: Deque[str] = deque(maxlen=6)
        self.last_status = "Ready"
        self.frame_history: Deque[FrameSample] = deque(maxlen=15)
        self.raw_landmarks: Deque[List[Tuple[float, float, float]]] = deque(
            maxlen=self.landmark_average_window
        )
        self.gesture_votes: Deque[str] = deque(maxlen=self.action_vote_window)
        self.cooldowns: Dict[str, float] = {}
        self.fsm: Dict[str, GestureFSM] = {
            name: GestureFSM()
            for name in ["palm", "point", "two_finger", "three_finger_click", "fist", "v_sign"]
        }

        self.cursor_mode = False
        self.task_view_active = False
        self.hand_missing_frames = 0
        self.last_pointer_screen: Optional[Point] = None
        self.last_nav_time = 0.0
        self.last_nav_tip: Optional[Point] = None

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def add_status(self, text: str) -> None:
        self.last_status = text
        self.overlay_lines.appendleft(text)

    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def vector(
        a: Tuple[float, float, float], b: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        return (b[0] - a[0], b[1] - a[1], b[2] - a[2])

    def angle(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        c: Tuple[float, float, float],
    ) -> float:
        ab = self.vector(b, a)
        cb = self.vector(b, c)
        dot = sum(x * y for x, y in zip(ab, cb))
        mag_ab = math.sqrt(sum(x * x for x in ab))
        mag_cb = math.sqrt(sum(x * x for x in cb))
        if mag_ab < 1e-6 or mag_cb < 1e-6:
            return 180.0
        cosine = self.clamp(dot / (mag_ab * mag_cb), -1.0, 1.0)
        return math.degrees(math.acos(cosine))

    def is_finger_extended(
        self,
        pts: List[Tuple[float, float, float]],
        tip: int,
        pip: int,
        mcp: int,
        wrist: int = 0,
    ) -> bool:
        tip_angle = self.angle(pts[mcp], pts[pip], pts[tip])
        lift = pts[pip][1] - pts[tip][1]
        reach = self.distance(
            (pts[tip][0], pts[tip][1]),
            (pts[wrist][0], pts[wrist][1]),
        )
        mid_reach = self.distance(
            (pts[pip][0], pts[pip][1]),
            (pts[wrist][0], pts[wrist][1]),
        )
        return tip_angle > 155.0 and lift > 0.02 and reach > (mid_reach * 1.08)

    def is_thumb_extended(self, pts: List[Tuple[float, float, float]]) -> bool:
        angle = self.angle(pts[1], pts[2], pts[4])
        base = self.distance((pts[2][0], pts[2][1]), (pts[5][0], pts[5][1]))
        span = self.distance((pts[4][0], pts[4][1]), (pts[2][0], pts[2][1]))
        lateral = abs(pts[4][0] - pts[3][0])
        return angle > 145.0 and span > base * 0.65 and lateral > 0.02

    def smooth_landmarks(
        self, landmarks: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        self.raw_landmarks.append(landmarks)
        count = len(self.raw_landmarks)
        smoothed: List[Tuple[float, float, float]] = []
        for idx in range(21):
            sx = sum(frame[idx][0] for frame in self.raw_landmarks) / count
            sy = sum(frame[idx][1] for frame in self.raw_landmarks) / count
            sz = sum(frame[idx][2] for frame in self.raw_landmarks) / count
            smoothed.append((sx, sy, sz))
        return smoothed

    def compute_confidence(self, handedness) -> float:
        if not handedness:
            return 0.0
        return float(handedness.classification[0].score)

    def classify_hand(
        self,
        pts: List[Tuple[float, float, float]],
        confidence: float,
        now: float,
    ) -> FrameSample:
        thumb = self.is_thumb_extended(pts)
        index = self.is_finger_extended(pts, 8, 6, 5)
        middle = self.is_finger_extended(pts, 12, 10, 9)
        ring = self.is_finger_extended(pts, 16, 14, 13)
        pinky = self.is_finger_extended(pts, 20, 18, 17)
        finger_state: FingerState = (
            1 if thumb else 0,
            1 if index else 0,
            1 if middle else 0,
            1 if ring else 0,
            1 if pinky else 0,
        )

        palm_center = (
            (pts[0][0] + pts[5][0] + pts[9][0] + pts[13][0] + pts[17][0]) / 5.0,
            (pts[0][1] + pts[5][1] + pts[9][1] + pts[13][1] + pts[17][1]) / 5.0,
        )
        hand_size_norm = max(
            self.distance((pts[0][0], pts[0][1]), (pts[9][0], pts[9][1])),
            self.distance((pts[5][0], pts[5][1]), (pts[17][0], pts[17][1])),
            1e-6,
        )
        hand_size_px = hand_size_norm * max(self.frame_width, self.frame_height)
        index_tip = (pts[8][0], pts[8][1])
        middle_tip = (pts[12][0], pts[12][1])
        thumb_tip = (pts[4][0], pts[4][1])

        pinch_distance = self.distance(index_tip, thumb_tip) / hand_size_norm
        v_spread = self.distance(index_tip, middle_tip) / hand_size_norm

        velocity = (0.0, 0.0)
        velocity_mag = 0.0
        horizontal_velocity = 0.0
        vertical_velocity = 0.0
        if self.frame_history:
            prev = self.frame_history[-1]
            dt = max(now - prev.timestamp, 1e-3)
            horizontal_velocity = (
                (palm_center[0] - prev.palm_center[0]) * self.frame_width / dt
            )
            vertical_velocity = (
                (palm_center[1] - prev.palm_center[1]) * self.frame_height / dt
            )
            velocity = (horizontal_velocity, vertical_velocity)
            velocity_mag = math.hypot(horizontal_velocity, vertical_velocity)

        gesture_flags = {
            "palm": finger_state == (1, 1, 1, 1, 1),
            "point": finger_state == (0, 1, 0, 0, 0),
            "two_finger": finger_state == (0, 1, 1, 0, 0) and v_spread < 0.38,
            "three_finger_click": finger_state == (0, 1, 1, 1, 0),
            "v_sign": finger_state == (0, 1, 1, 0, 0) and v_spread >= 0.38,
            "fist": finger_state == (0, 0, 0, 0, 0),
        }

        return FrameSample(
            timestamp=now,
            palm_center=palm_center,
            index_tip=index_tip,
            middle_tip=middle_tip,
            hand_size_px=hand_size_px,
            finger_state=finger_state,
            gesture_flags=gesture_flags,
            velocity=velocity,
            velocity_mag=velocity_mag,
            horizontal_velocity=horizontal_velocity,
            vertical_velocity=vertical_velocity,
            pinch_distance=pinch_distance,
            v_spread=v_spread,
            confidence=confidence,
        )

    def update_fsm(self, name: str, active: bool, now: float) -> GestureFSM:
        state = self.fsm[name]
        if active:
            if not state.active:
                state.entered_at = now
            state.active = True
        else:
            state.active = False
            state.entered_at = 0.0
            state.meta.clear()
        return state

    def can_fire(self, key: str, now: float) -> bool:
        return now >= self.cooldowns.get(key, 0.0)

    def set_cooldown(self, key: str, seconds: float, now: float) -> None:
        self.cooldowns[key] = now + seconds

    def safe_action(self, label: str, fn) -> bool:
        try:
            fn()
            self.add_status(label)
            return True
        except Exception as exc:
            self.add_status(f"Action error: {exc}")
            return False

    def map_to_screen(self, point: Point) -> Point:
        margin_x = 0.18
        margin_y = 0.20
        x = (point[0] - margin_x) / (1.0 - 2.0 * margin_x)
        y = (point[1] - margin_y) / (1.0 - 2.0 * margin_y)
        x = self.clamp(x, 0.0, 1.0)
        y = self.clamp(y, 0.0, 1.0)
        sx = x * self.screen_w
        sy = y * self.screen_h
        if self.last_pointer_screen is None:
            smoothed = (sx, sy)
        else:
            alpha = 0.28
            smoothed = (
                self.last_pointer_screen[0] * (1.0 - alpha) + sx * alpha,
                self.last_pointer_screen[1] * (1.0 - alpha) + sy * alpha,
            )
        self.last_pointer_screen = smoothed
        return smoothed

    def reset_modes(self) -> None:
        self.cursor_mode = False
        self.last_pointer_screen = None
        self.last_nav_tip = None

    def handle_palm(self, sample: FrameSample, now: float) -> None:
        state = self.update_fsm(
            "palm",
            sample.gesture_flags["palm"]
            and sample.confidence >= self.min_confidence
            and sample.velocity_mag < 120,
            now,
        )
        if (
            state.active
            and now - state.entered_at >= 0.55
            and self.can_fire("desktop_toggle", now)
        ):
            if self.safe_action("Desktop toggle", lambda: pyautogui.hotkey("win", "d")):
                self.set_cooldown("desktop_toggle", 1.0, now)
                state.entered_at = now + 999.0

    def handle_point_task_view(self, sample: FrameSample, now: float) -> None:
        state = self.update_fsm(
            "point",
            sample.gesture_flags["point"] and sample.confidence >= self.min_confidence,
            now,
        )
        if not state.active:
            self.last_nav_tip = None
            return

        if (
            not self.task_view_active
            and now - state.entered_at >= 0.30
            and self.can_fire("task_view", now)
        ):
            if self.safe_action("Task view", lambda: pyautogui.hotkey("win", "tab")):
                self.task_view_active = True
                self.cursor_mode = False
                self.last_nav_tip = sample.index_tip
                self.set_cooldown("task_view", 1.0, now)
                return

        if not self.task_view_active:
            return

        if self.last_nav_tip is None:
            self.last_nav_tip = sample.index_tip
            return

        if now - self.last_nav_time < 0.14:
            return

        dx = sample.index_tip[0] - self.last_nav_tip[0]
        dy = sample.index_tip[1] - self.last_nav_tip[1]
        moved = False

        if abs(dx) >= abs(dy) and abs(dx) > 0.06:
            key = "right" if dx > 0 else "left"
            moved = self.safe_action(f"Task view {key}", lambda: pyautogui.press(key))
        elif abs(dy) > abs(dx) and abs(dy) > 0.08:
            key = "down" if dy > 0 else "up"
            moved = self.safe_action(f"Task view {key}", lambda: pyautogui.press(key))

        if moved:
            self.last_nav_tip = sample.index_tip
            self.last_nav_time = now

    def handle_two_finger_cursor(self, sample: FrameSample, now: float) -> None:
        state = self.update_fsm(
            "two_finger",
            sample.gesture_flags["two_finger"] and sample.confidence >= self.min_confidence,
            now,
        )
        if not state.active:
            if (
                self.cursor_mode
                and not self.task_view_active
                and not sample.gesture_flags["three_finger_click"]
            ):
                self.cursor_mode = False
            return

        if now - state.entered_at >= 0.20:
            self.cursor_mode = True
            cursor_tip = (
                (sample.index_tip[0] + sample.middle_tip[0]) / 2.0,
                (sample.index_tip[1] + sample.middle_tip[1]) / 2.0,
            )
            screen_point = self.map_to_screen(cursor_tip)
            pyautogui.moveTo(screen_point[0], screen_point[1], _pause=False)
            self.add_status("Cursor mode")

    def handle_three_finger_click(self, sample: FrameSample, now: float) -> None:
        state = self.update_fsm(
            "three_finger_click",
            sample.gesture_flags["three_finger_click"]
            and sample.confidence >= self.min_confidence
            and self.cursor_mode,
            now,
        )
        if not state.active:
            return

        cursor_tip = (
            (sample.index_tip[0] + sample.middle_tip[0]) / 2.0,
            (sample.index_tip[1] + sample.middle_tip[1]) / 2.0,
        )
        screen_point = self.map_to_screen(cursor_tip)
        pyautogui.moveTo(screen_point[0], screen_point[1], _pause=False)

        if now - state.entered_at >= 0.18 and self.can_fire("cursor_click", now):
            if self.safe_action("Cursor click", pyautogui.click):
                self.set_cooldown("cursor_click", 0.5, now)
                state.entered_at = now + 999.0

    def handle_fist(self, sample: FrameSample, now: float) -> None:
        state = self.update_fsm(
            "fist",
            sample.gesture_flags["fist"]
            and sample.confidence >= self.min_confidence
            and sample.velocity_mag < 100,
            now,
        )
        if not state.active or now - state.entered_at < 0.35:
            return

        if self.task_view_active and self.can_fire("task_select", now):
            if self.safe_action("Open selected app", lambda: pyautogui.press("enter")):
                self.task_view_active = False
                self.last_nav_tip = None
                self.set_cooldown("task_select", 0.8, now)
                state.entered_at = now + 999.0

    def handle_v_sign(self, sample: FrameSample, now: float) -> None:
        state = self.update_fsm(
            "v_sign",
            sample.gesture_flags["v_sign"]
            and sample.confidence >= self.min_confidence
            and sample.velocity_mag < 110,
            now,
        )
        if (
            state.active
            and now - state.entered_at >= 0.45
            and self.can_fire("close_app", now)
        ):
            if self.safe_action("Close current app", lambda: pyautogui.hotkey("alt", "f4")):
                self.set_cooldown("close_app", 1.0, now)
                state.entered_at = now + 999.0

    def build_vote_label(self, sample: FrameSample) -> str:
        active_labels = [name for name, active in sample.gesture_flags.items() if active]
        return "+".join(active_labels) if active_labels else "none"

    def draw_overlay(self, frame, sample: Optional[FrameSample], fps: float) -> None:
        panel = frame.copy()
        cv2.rectangle(panel, (12, 12), (390, 230), (25, 25, 25), -1)
        frame[:] = cv2.addWeighted(panel, 0.30, frame, 0.70, 0)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 180),
            2,
        )
        cv2.putText(
            frame,
            f"Status: {self.last_status}",
            (24, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Task View: {'ON' if self.task_view_active else 'OFF'}",
            (24, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Cursor: {'ON' if self.cursor_mode else 'OFF'}",
            (24, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
        )

        if sample is not None:
            cv2.putText(
                frame,
                f"Fingers: {sample.finger_state}",
                (24, 148),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Vel px/s: {sample.velocity_mag:.0f}",
                (24, 176),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Spread: {sample.v_spread:.2f}",
                (24, 204),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            index_px = (
                int(sample.index_tip[0] * self.frame_width),
                int(sample.index_tip[1] * self.frame_height),
            )
            middle_px = (
                int(sample.middle_tip[0] * self.frame_width),
                int(sample.middle_tip[1] * self.frame_height),
            )
            palm_px = (
                int(sample.palm_center[0] * self.frame_width),
                int(sample.palm_center[1] * self.frame_height),
            )
            cv2.circle(frame, index_px, 10, (0, 255, 255), 2)
            cv2.circle(frame, middle_px, 8, (255, 180, 0), 2)
            cv2.circle(frame, palm_px, 14, (0, 128, 255), 2)

        y = 258
        for line in list(self.overlay_lines)[:5]:
            cv2.putText(
                frame,
                line,
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
            )
            y += 22

    def run(self) -> None:
        if not self.cap.isOpened():
            print("Error: could not open webcam.")
            return

        window_name = "Hand Gesture Recognition and Action Control System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        prev_time = time.time()
        fps = 0.0
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    self.add_status("Camera frame unavailable")
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue

                frame = cv2.flip(frame, 1)
                self.frame_height, self.frame_width = frame.shape[:2]
                now = time.time()
                dt = max(now - prev_time, 1e-6)
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
                prev_time = now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.hands.process(rgb)
                rgb.flags.writeable = True

                sample: Optional[FrameSample] = None
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness = results.multi_handedness[0] if results.multi_handedness else None
                    confidence = self.compute_confidence(handedness)
                    if confidence >= self.min_confidence:
                        raw_points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        smoothed = self.smooth_landmarks(raw_points)
                        sample = self.classify_hand(smoothed, confidence, now)
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(
                                color=(0, 255, 180),
                                thickness=2,
                                circle_radius=2,
                            ),
                            self.mp_draw.DrawingSpec(
                                color=(255, 255, 255),
                                thickness=2,
                                circle_radius=2,
                            ),
                        )
                        self.hand_missing_frames = 0
                    else:
                        self.add_status("Low confidence hand tracking")
                        self.hand_missing_frames += 1
                else:
                    self.hand_missing_frames += 1

                if sample is not None:
                    self.frame_history.append(sample)
                    self.gesture_votes.append(self.build_vote_label(sample))
                    self.handle_palm(sample, now)
                    self.handle_point_task_view(sample, now)
                    self.handle_two_finger_cursor(sample, now)
                    self.handle_three_finger_click(sample, now)
                    self.handle_fist(sample, now)
                    self.handle_v_sign(sample, now)
                elif self.hand_missing_frames > 3:
                    self.raw_landmarks.clear()
                    self.reset_modes()
                    for name in self.fsm:
                        self.update_fsm(name, False, now)

                vote_text = (
                    Counter(self.gesture_votes).most_common(1)[0][0]
                    if self.gesture_votes
                    else "none"
                )
                cv2.putText(
                    frame,
                    f"Vote: {vote_text}",
                    (410, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                )
                self.draw_overlay(frame, sample, fps)
                display_frame = frame
                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
                    if win_w > 0 and win_h > 0:
                        display_frame = cv2.resize(frame, (win_w, win_h))
                except cv2.error:
                    pass
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.hands.close()
            except Exception:
                pass
            self.cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    GestureController().run()


if __name__ == "__main__":
    main()
