
from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    id: str
    class_label: Optional[bool]
