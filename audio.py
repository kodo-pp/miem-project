# -*- coding: utf-8 -*-

from pyaudio import PyAudio, paInt16


class AudioRecorder:
    def __init__(self, chunk_size=1024, rate=44100):
        self.pa = PyAudio()
        self.stream = self.pa.open(
            format            = paInt16,
            channels          = 1,
            rate              = rate,
            input             = True,
            frames_per_buffer = chunk_size
        )
        self.stream.stop_stream()
        self.chunk_size = chunk_size
        self.rate = rate
        self.format = paInt16

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.close()

    def record(self, seconds):
        data = b''
        self.stream.start_stream()
        for i in range(int(self.rate / self.chunk_size * seconds)):
            frame = self.stream.read(self.chunk_size)
            data += frame
        self.stream.stop_stream()
        return data

    def bytes_to_numseq(self, b):
        size = self.pa.get_sample_size(self.format)
        i = 0
        while i < len(b):
            yield int.from_bytes(b[i : i+size], signed=True, byteorder='little')
            i += size

    def close(self):
        self.input_audio_stream.close()
        self.pa.terminate()
