# Hand Gesture Recognition - BCA Major Project 2026

This project uses OpenCV + MediaPipe Hands for real-time hand detection and gesture-driven desktop control, with stronger temporal smoothing and safer desktop targeting.

## Features

- Robust gesture recognition with hand steadiness checks, frame voting, and cooldowns
- Safe external-app targeting so destructive gestures never close this app itself
- Open Task View and navigate it with hand motion
- Minimize an app, show desktop, or close multiple apps with dedicated gestures
- Modular and scalable project structure
- Visual mode tinting by gesture family

## Gesture Mapping

- `Scissors / V sign (index + middle spread apart)` -> Cut the selected external app
- `Point (index only)` -> Open Task View
- `Move pointing finger while Task View is open` -> Navigate left / right / up / down
- `Fist` -> Select the highlighted window from Task View
- `Three fingers (index + middle + ring)` -> Minimize the selected external app
- `Thumbs up` -> Show desktop
- `Rock sign (index + pinky)` -> Close multiple external apps in a bounded safe loop

Notes:
- Gestures trigger only after they stay stable, win the recent frame vote, and the hand is physically steady.
- The `cut` gesture only targets other apps. This app protects itself and will not close its own window.
- For best results, briefly focus the app you want to control before showing a gesture.
- `Close multiple apps` is still best-effort and intentionally bounded by a fixed iteration limit.

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

## Configuration

Tune runtime behavior in `hand_gesture/config.py`:

- `consecutive_frames_required`
- `action_vote_window`
- `action_vote_ratio`
- `action_cooldown_seconds`
- `hand_steady_delta`
- `steady_frames_required`
- `close_all_iterations`
- `close_all_step_delay_seconds`
