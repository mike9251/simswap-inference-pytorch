from enum import Enum


class CheckpointType(Enum):
    OFFICIAL_224 = "official_224"
    UNOFFICIAL = "none"


class FaceAlignmentType(Enum):
    FFHQ = "ffhq"
    DEFAULT = "none"
