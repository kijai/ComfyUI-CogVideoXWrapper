NUM_FRAMES = None
FETA_WEIGHT = None
ENABLE_FETA = False

def set_num_frames(num_frames: int):
    global NUM_FRAMES
    NUM_FRAMES = num_frames


def get_num_frames() -> int:
    return NUM_FRAMES


def enable_enhance():
    global ENABLE_FETA
    ENABLE_FETA = True

def disable_enhance():
    global ENABLE_FETA
    ENABLE_FETA = False

def is_enhance_enabled() -> bool:
    return ENABLE_FETA

def set_enhance_weight(feta_weight: float):
    global FETA_WEIGHT
    FETA_WEIGHT = feta_weight


def get_enhance_weight() -> float:
    return FETA_WEIGHT
