# -*- coding: utf-8 -*-

import config as conf

def normalize(arr):
    minval = min(arr)
    maxval = max(arr)
    if minval == maxval:
        return [0.5] * len(arr)
    return [(v - minval) / (maxval - minval) for v in arr]

def normalize_scalar(v, minval, maxval):
    if minval == maxval:
        return 0
    return v / (maxval - minval)

def arr2diff(arr):
    m = len(arr)
    return [abs(arr[i] - arr[i - 1]) for i in range(1, m)]

def sliding_diff(arr, winsize):
    # import pudb; pudb.set_trace()
    m = len(arr)
    if not (0 < winsize <= m):
        raise ValueError('winsize should be between 0 and len(arr)')
    d = arr2diff(arr)
    accum = sum(d[:winsize-1])
    amaxv = max(arr[:winsize])
    aminv = min(arr[:winsize])
    yield normalize_scalar(accum, aminv, amaxv) / (winsize - 1)
    for i in range(m - winsize):
        accum -= d[i]
        accum += d[i + winsize - 1]

        if arr[i + winsize] < aminv:
            aminv = arr[i + winsize]
        elif arr[i] == aminv:
            aminv = min(arr[i+1 : i+winsize+1])

        if arr[i + winsize] > amaxv:
            amaxv = arr[i + winsize]
        elif arr[i] == aminv:
            amaxv = max(arr[i+1 : i+winsize+1])
        yield normalize_scalar(accum, aminv, amaxv) / (winsize - 1)
