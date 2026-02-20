# Hand Gesture Recognition - BCA Major Project 2026

This project uses OpenCV + MediaPipe Hands for real-time hand detection and gesture-driven desktop control.

## Features

- Close current app using hand gesture
- Switch to another window using hand gesture
- Close multiple apps using hand gesture (bounded safe loop)
- Modular and scalable project structure
- Visual color effects by finger count
- Stable-frame detection and cooldown to reduce accidental triggers

## Gesture Mapping

- `Open palm (5 fingers)` -> Close current app
- `Fist (0 fingers)` -> Switch window
- `V sign (index + middle)` -> Close multiple apps (best effort)

Notes:
- Gestures trigger only after staying stable for multiple frames.
- Cooldown prevents repeated accidental execution.

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
- `action_cooldown_seconds`
- `close_all_iterations`
- `close_all_step_delay_seconds`
