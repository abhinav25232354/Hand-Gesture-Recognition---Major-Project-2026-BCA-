from __future__ import annotations

import ctypes
import os
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
        self._self_pid = os.getpid()
        self._last_external_hwnd: Optional[int] = None
        self._task_view_active = False

        if pyautogui is not None:
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.05

    def refresh_external_target(self) -> None:
        if self._task_view_active:
            return
        if self.os_name != "windows":
            return
        hwnd = _get_foreground_window()
        if hwnd is None:
            return
        if _get_window_pid(hwnd) == self._self_pid:
            return
        self._last_external_hwnd = hwnd

    def execute(self, action: GestureAction) -> bool:
        self.last_error = None
        if self.os_name != "windows" and pyautogui is None:
            self.last_error = "pyautogui not installed. Run: pip install pyautogui"
            return False

        try:
            if action == GestureAction.CLOSE_CURRENT_APP:
                self._close_current_app()
            elif action == GestureAction.OPEN_TASK_VIEW:
                self._open_task_view()
            elif action == GestureAction.SELECT_TASK_WINDOW:
                self._select_task_view_window()
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
            if not self._close_last_external_window():
                self.last_error = "No external app selected. Focus the app you want to control first."
                raise RuntimeError(self.last_error)
        elif self.os_name == "darwin":
            pyautogui.hotkey("command", "q")
        else:
            pyautogui.hotkey("alt", "f4")

    def _switch_window(self) -> None:
        if self.os_name == "windows":
            if not self._focus_last_external_window():
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
            self.refresh_external_target()

    @property
    def task_view_active(self) -> bool:
        return self._task_view_active

    def _open_task_view(self) -> None:
        if self.os_name == "windows":
            if self._task_view_active:
                return
            _send_windows_hotkey("win", "tab")
            self._task_view_active = True
            return

        # Fallback to existing switch behavior on non-Windows.
        self._switch_window()

    def navigate_task_view(self, direction: str) -> bool:
        if self.os_name != "windows" or not self._task_view_active:
            return False
        direction = direction.lower()
        if direction not in {"left", "right", "up", "down"}:
            return False
        _send_windows_hotkey(direction)
        return True

    def _select_task_view_window(self) -> None:
        if self.os_name == "windows":
            if not self._task_view_active:
                self.last_error = "Task View is not active."
                raise RuntimeError(self.last_error)
            _send_windows_hotkey("enter")
            self._task_view_active = False
            return
        self._switch_window()

    def _focus_last_external_window(self) -> bool:
        hwnd = self._last_external_hwnd
        if self.os_name != "windows" or hwnd is None:
            return False
        if not _is_window_alive(hwnd):
            self._last_external_hwnd = None
            return False
        return _focus_window(hwnd)

    def _close_last_external_window(self) -> bool:
        hwnd = self._last_external_hwnd
        if self.os_name != "windows" or hwnd is None:
            return False
        if not _is_window_alive(hwnd):
            self._last_external_hwnd = None
            return False
        if not _focus_window(hwnd):
            return False
        time.sleep(0.05)
        _send_windows_hotkey("alt", "f4")
        return True


_WINDOWS_VK = {
    "win": 0x5B,
    "alt": 0x12,
    "tab": 0x09,
    "f4": 0x73,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "enter": 0x0D,
}
_KEYEVENTF_KEYUP = 0x0002
_SW_RESTORE = 9


def _get_foreground_window() -> Optional[int]:
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    return int(hwnd) if hwnd else None


def _get_window_pid(hwnd: int) -> int:
    pid = ctypes.c_ulong(0)
    ctypes.windll.user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(pid))
    return int(pid.value)


def _is_window_alive(hwnd: int) -> bool:
    return bool(ctypes.windll.user32.IsWindow(ctypes.c_void_p(hwnd)))


def _focus_window(hwnd: int) -> bool:
    user32 = ctypes.windll.user32
    user32.ShowWindow(ctypes.c_void_p(hwnd), _SW_RESTORE)
    return bool(user32.SetForegroundWindow(ctypes.c_void_p(hwnd)))


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
