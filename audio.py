# -*- coding: utf-8 -*-

import os
import sys

from pyaudio import PyAudio, paInt16

import config as conf

class AudioRecorder:
    def __init__(self, rate):
        self.pa = PyAudio()
        self.stream = self.pa.open(
            format            = paInt16,
            channels          = 1,
            rate              = rate,
            input             = True,
            frames_per_buffer = conf.block_size
        )
        self.stream.stop_stream()
        self.rate = rate
        self.format = paInt16

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.close()

    def record(self, length):
        self.stream.start_stream()
        data = self.stream.read(length)
        self.stream.stop_stream()
        return data

    def bytes_to_numseq(self, b):
        size = self.pa.get_sample_size(self.format)
        i = 0
        while i < len(b):
            yield int.from_bytes(b[i : i+size], signed=True, byteorder='little')
            i += size

    def close(self):
        self.stream.close()
        self.pa.terminate()
