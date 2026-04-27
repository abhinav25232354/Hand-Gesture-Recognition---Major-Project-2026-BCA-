# Hand Gesture Recognition & Action Control System - BCA Major Project 2026 (Finished)

This project uses OpenCV + MediaPipe Hands for real-time hand detection and gesture-driven desktop control on Windows 11. The system is tuned for a smaller, more predictable gesture set so interaction feels smoother and easier to learn.

## Features

- Real-time webcam control at `640x480`
- MediaPipe Hands with `static_image_mode=False`, `model_complexity=1`, and `max_num_hands=1`
- Smoothed landmark tracking with multi-frame averaging
- Gesture stability using hold time, cooldowns, and finite-state transitions
- Mouse-free desktop interaction with `pyautogui`
- Task View navigation using fingertip motion
- On-screen overlays for status, gesture state, and finger positions
- Fallback handling for low-confidence tracking and temporary hand loss

## Gesture Mapping

- `Open palm (hold)` -> Toggle desktop using `Win + D`
  - If apps are visible, it shows the desktop.
  - If the desktop is already visible, showing the palm again restores the previous windows.
- `Index finger point (hold)` -> Open Task View using `Win + Tab`
- `Move index finger while Task View is open` -> Navigate between apps and windows in Task View
- `Make a fist while Task View is open` -> Open the selected app/window
- `Index + middle finger together` -> Enable direct cursor control without opening Task View
- `Keep using index + middle fingers` -> Move the cursor naturally on screen
- `Open the third finger while cursor mode is active` -> Left click at the current cursor position
- `V-sign (index + middle spread apart)` -> Close the current app using `Alt + F4`

## How It Works

The system processes MediaPipe landmarks by:

- Normalizing landmark coordinates to the frame
- Detecting finger extension using joint-angle checks
- Measuring finger spread to separate two-finger cursor mode from the V-sign
- Detecting the third-finger extension as a click command during cursor mode
- Averaging recent landmark frames to reduce jitter
- Applying confidence thresholds and short hold times for reliable activation
- Tracking fingertip movement to navigate Task View naturally

## Project Structure

```text
.
|-- main.py
|-- hand_gesture/
|   |-- __init__.py
|   |-- actions.py
|   |-- config.py
|   |-- controller.py
|   |-- effects.py
|   |-- gestures.py
|   |-- ui.py
|   `-- vision.py
|-- requirements.txt
`-- README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Press `q` to quit.

## Notes

- Use one hand at a time for the most stable tracking.
- Keep lighting even and avoid strong backlight for better confidence.
- Hold the palm, point, or V-sign briefly so the stabilizer can confirm the gesture.
- Two fingers close together activate cursor mode.
- Extending the third finger during cursor mode performs a click.
- Two fingers spread apart act as the V-sign and close the active app.
- The project is optimized for Windows 11 desktop control.
