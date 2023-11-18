import random
from typing import Optional

def truncnorm(mean: float, std: float, min: Optional[float], max: Optional[float]) -> float:
    '''Truncated normal distribution'''
    result = random.normalvariate(mean, std)
    if min != None and max != None:
        return result if result >= min and result <= max else truncnorm(mean, std, min, max)
    elif min != None:
        return result if result >= min else truncnorm(mean, std, min, max)
    elif max != None:
        return result if result <= max else truncnorm(mean, std, min, max)
    else:
        return result