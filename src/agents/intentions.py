"""Intention vocabulary. Goals are spatial targets; intentions are
short-horizon path strategies subordinate to the current goal."""

from enum import Enum


class Intention(str, Enum):
    DIRECT = "DIRECT"
    DETOUR_LEFT = "DETOUR_LEFT"
    DETOUR_RIGHT = "DETOUR_RIGHT"
    WAIT = "WAIT"


ALL_INTENTIONS = [i.value for i in Intention]
