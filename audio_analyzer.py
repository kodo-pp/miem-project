# -*- coding: utf-8 -*-

from threading import Thread, Lock
from queue import Queue

class AudioAnalyzer:
    def __init__(self, recorder, time_quant=0.2, queue_size=441000):
        self.recorder = recorder
        self.time_quant = time_quant
        self.data = Queue(maxsize=queue_size)
        self.lock = Lock()
        self.do_quit = False

    def analyze(self):
        recorder_thread = Thread(target=self._record_func)
        recorder_thread.start()

        self.counter = 0
        max_qsize = 0
        try:
            print()
            i = 0
            while True:
                i += 1
                elem = self.data.get(timeout = self.time_quant * 5)
                self.counter += 1
                qsize = self.data.qsize()
                max_qsize = max(max_qsize, qsize)
                if i % 100 == 0:
                    print(
                        "\x1b[A"                                    # Подняться на предыдущую строку теминала
                        "\x1b[2K"                                   # Очистить строку
                        "\x1b[0;1m" "counter = \x1b[0;35m{}" "\t"   # Счётчик чисел
                        "\x1b[0;1m" "qsize = \x1b[0;35m{}" "\t"     # Текущий размер очереди
                        "\x1b[0;1m" "max_qsize = \x1b[0;35m{}"      # Максимальный размер очереди
                        "\x1b[0m".format(self.counter, qsize, max_qsize)
                    )
        finally:
            print('\x1b[0m', end='')
            print('Exiting...')
            with self.lock:
                self.do_quit = True
            recorder_thread.join()

    def _record_func(self):
        while True:
            for i in self.recorder.record(self.time_quant):
                self.data.put(i)
            with self.lock:
                if self.do_quit:
                    return
