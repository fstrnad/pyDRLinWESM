"""
This enumeration is used to differ between the three possible outcomes in our model.
1. The agent can leave the planetary boundaries (OUT_PB)
2. It can stay within the boundaries but in an unsustainable state (BROWN_FP)
3. It can stay within the boundaries and within a sustainable state (GREEN_FP)

@author: Felix Strnad
"""
from enum import Enum
class Basins(Enum):
    OUT_PB=0
    BROWN_FP=1
    GREEN_FP=2