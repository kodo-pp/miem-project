# -*- coding: utf-8 -*-

import config as conf

def format_string(fmt):
    known_formats = {
        'black': '30',
        'red': '31',
        'green': '32',
        'brown': '33',
        'blue': '34',
        'magenta': '35',
        'purple': '35',
        'cyan': '36',
        'white': '37',
        'bold': '1',
    }
    return known_formats[fmt]

def format(text, fmt):
    # Не пугаться, всё нормально
    return '\x1b[{}m{}\x1b[0m'.format(format_string(fmt), text)
