import random
from typing import Optional

def truncnorm(mean: float, std: float, min: Optional[float], max: Optional[float]) -> float:
    '''Truncated normal distribution'''
    result = random.normalvariate(mean, std)

    if min != None and max != None:
        while result < min or result > max:
            result = random.normalvariate(mean, std)
    elif min != None:
        while result < min:
            result = random.normalvariate(mean, std)
    elif max != None:
        while result > max:
            result = random.normalvariate(mean, std)

    return result

def truncexp(mean: float, min: Optional[float], max: Optional[float]) -> float:
    '''Truncated exponential distribution'''
    result = random.expovariate(1/mean)

    if min != None and max != None:
        if min >= max:
            raise ValueError(f"min ({min}) must be smaller than max ({max})!")
        while result < min or result > max:
            result = random.expovariate(1/mean)
    elif min != None:
        while result < min:
            result = random.expovariate(1/mean)
    elif max != None:
        while result > max:
            result = random.expovariate(1/mean)

    return result

