# -*- coding: utf-8 -*-

from threading import Thread, Lock
from queue import Queue

from linear_regression import LinearRegression

class AudioAnalyzer:
    def __init__(self, recorder, frame_length=4000, queue_size=441000):
        self.recorder = recorder
        self.frame_length = frame_length
        self.data = Queue(maxsize=queue_size)
        self.lock = Lock()
        self.do_quit = False

    def analyze(self):
        recorder_thread = Thread(target=self._record_func)
        recorder_thread.start()

        max_qsize = 0
        try:
            i = 0
            block = []
            block_step = 40
            block_length = 4000
            max_qsize = 0
            lr = LinearRegression(block_length)
            try:
                with open('learnt.txt') as rf:
                    print('Reading weights from learnt.txt')
                    weights = [float(i) for i in rf.read().split(' ')]
            except FileNotFoundError:
                weights = [1] * (block_length + 1)
            lr.set_weights(weights)
            print()
            while True:
                i += block_step
                while len(block) < block_length:
                    block.append(self.data.get(timeout=5))
                qsize = self.data.qsize()
                max_qsize = max(max_qsize, qsize)
                output = lr.get(block)
                block = block[block_step:]

                if i % 100 == 0:
                    print(
                        "\x1b[A"                                    # Подняться на предыдущую строку теминала
                        "\x1b[2K"                                   # Очистить строку
                        "\x1b[0;1m" "counter = \x1b[0;35m{}" "\t"   # Счётчик чисел
                        "\x1b[0;1m" "qsize = \x1b[0;35m{}" "\t"     # Текущий размер очереди
                        "\x1b[0;1m" "max_qsize = \x1b[0;35m{}" "\t" # Максимальный размер очереди
                        "\x1b[0;1m" "output = \x1b[0;35m{}"         # Предсказанное значение
                        "\x1b[0m".format(i, qsize, max_qsize, output)
                    )
        finally:
            print('\x1b[0m', end='')
            print('Exiting...')
            with self.lock:
                self.do_quit = True
            recorder_thread.join()

    def _record_func(self):
        while True:
            for i in self.recorder.record(self.frame_length):
                self.data.put(i)
            with self.lock:
                if self.do_quit:
                    return
