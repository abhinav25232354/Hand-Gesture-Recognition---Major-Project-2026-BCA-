from __future__ import annotations

import ctypes
import platform
import time
from typing import Optional

from hand_gesture.gestures import GestureAction

try:
    import pyautogui
except Exception:
    pyautogui = None


class DesktopActionExecutor:
    def __init__(self, close_all_iterations: int, close_all_step_delay_seconds: float):
        self.close_all_iterations = close_all_iterations
        self.close_all_step_delay_seconds = close_all_step_delay_seconds
        self.os_name = platform.system().lower()
        self.last_error: Optional[str] = None

        if pyautogui is not None:
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.05

    def execute(self, action: GestureAction) -> bool:
        self.last_error = None
        if self.os_name != "windows" and pyautogui is None:
            self.last_error = "pyautogui not installed. Run: pip install pyautogui"
            return False

        try:
            if action == GestureAction.CLOSE_CURRENT_APP:
                self._close_current_app()
            elif action == GestureAction.SWITCH_WINDOW:
                self._switch_window()
            elif action == GestureAction.CLOSE_ALL_APPS:
                self._close_all_apps()
            else:
                return False
            return True
        except Exception as ex:
            self.last_error = f"Action failed: {ex}"
            return False

    def _close_current_app(self) -> None:
        if self.os_name == "windows":
            _send_windows_hotkey("alt", "f4")
        elif self.os_name == "darwin":
            pyautogui.hotkey("command", "q")
        else:
            pyautogui.hotkey("alt", "f4")

    def _switch_window(self) -> None:
        if self.os_name == "windows":
            _send_windows_hotkey("alt", "tab")
        elif self.os_name == "darwin":
            pyautogui.hotkey("command", "tab")
        else:
            pyautogui.hotkey("alt", "tab")

    def _close_all_apps(self) -> None:
        for _ in range(self.close_all_iterations):
            self._close_current_app()
            time.sleep(self.close_all_step_delay_seconds)
            self._switch_window()
            time.sleep(self.close_all_step_delay_seconds)


_WINDOWS_VK = {
    "alt": 0x12,
    "tab": 0x09,
    "f4": 0x73,
}
_KEYEVENTF_KEYUP = 0x0002


def _send_windows_hotkey(*keys: str) -> None:
    user32 = ctypes.windll.user32
    vk_codes = []
    for key in keys:
        code = _WINDOWS_VK.get(key.lower())
        if code is None:
            raise ValueError(f"Unsupported Windows key: {key}")
        vk_codes.append(code)

    for code in vk_codes:
        user32.keybd_event(code, 0, 0, 0)
    for code in reversed(vk_codes):
        user32.keybd_event(code, 0, _KEYEVENTF_KEYUP, 0)
