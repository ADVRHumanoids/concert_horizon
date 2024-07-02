from enum import Enum
class OperationMode(Enum):
    IDLE = 0
    FOLLOW_ME = 1
    TEACH = 2
    HYBRID = 3
    HOMING = 4