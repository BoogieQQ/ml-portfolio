from enum import Enum


class __ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return [e.value for e in cls]
