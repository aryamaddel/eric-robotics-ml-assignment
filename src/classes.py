"""Single source of truth for the detector's classes."""

CLASSES = ["cone", "barrier", "stop_sign"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
DISPLAY = {"cone": "Cone", "barrier": "Barrier", "stop_sign": "Stop Sign"}
