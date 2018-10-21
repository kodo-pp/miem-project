# -*- coding: utf-8 -*-

import os
import sys
from threading import Thread, Lock

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

    def start_recording(self):
        self.lock = Lock()
        self._do_stop_recording = False
        def record_func(self):
            data = b''
            while True:
                with self.lock:
                    if self._do_stop_recording:
                        break
                data += self.record(conf.block_size)
            with self.lock:
                self._do_stop_recording = False
            self._recorded_data = data
        self._recorder_thread = Thread(target=record_func, args=(self,))
        self._recorder_thread.daemon = True
        self._recorder_thread.start()

    def finish_recording(self):
        sys.stdout.flush()
        with self.lock:
            sys.stdout.flush()
            self._do_stop_recording = True
        sys.stdout.flush()
        self._recorder_thread.join()
        sys.stdout.flush()
        return self._recorded_data

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
