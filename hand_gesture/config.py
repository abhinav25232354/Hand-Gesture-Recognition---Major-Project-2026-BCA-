from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    camera_index: int = 0
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    consecutive_frames_required: int = 8
    action_cooldown_seconds: float = 2.0
    close_all_iterations: int = 7
    close_all_step_delay_seconds: float = 0.2
