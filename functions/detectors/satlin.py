def satlin(interval):
    if 0 < interval < 1:
        return interval
    elif interval >= 1:
        return 1
    else:
        return 0