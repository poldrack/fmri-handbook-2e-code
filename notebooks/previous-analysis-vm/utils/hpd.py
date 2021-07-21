import numpy

def hpd(data,pct):
    """
    return indices for highest posterior density from array
    """
    assert numpy.sum(data)==1.0
    idx=numpy.argsort(data)[::-1]
    sorted_data=data[idx]

    return numpy.where(numpy.cumsum(sorted_data)<=pct)[0]
