# -*- coding: utf-8 -*-

import config as conf

def normalize(arr):
    minval = min(arr)
    maxval = max(arr)
    if minval == maxval:
        return [0.5] * len(arr)
    return [(v - minval) / (maxval - minval) for v in arr]

def arr2diff(arr):
    m = len(arr)
    return [abs(arr[i] - arr[i - 1]) for i in range(1, m)]

def sliding_diff(arr, winsize):
    m = len(arr)
    if not (0 < winsize <= m):
        raise ValueError('winsize should be between 0 and len(arr)')
    for i in range(m - winsize + 1):
        window = arr[i:i+winsize]
        norm = normalize(window)
        trans = arr2diff(norm)
        yield sum(trans) / winsize
