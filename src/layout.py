import numpy as np

import config
from prime import sample_primed


def wrap(text, max_chars=config.MAX_CHARS):
    lines, cur = [], ''

    for w in text.split():
        # hard split if the word is too long
        while len(w) > max(max_chars,20):
            # if a single word is longer than the hwole line and the hard split breaks it so it finishes that line then cut it at max character as a line of its own
            if cur:
                lines.append(cur) 
                cur = ''
            
            lines.append(w[:max_chars])
            w = w[max_chars:]
        # if line empty then the word starts it
        if not cur:
            cur = w
        # if line + space + word fits the max character then add that word to the line
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += ' ' + w
        # if it doesnt fit then finish that line and start a new one
        else:
            lines.append(cur)
            cur = w
    # add remaining unifnished line            
    if cur:
        lines.append(cur)
    return lines

def generate_best(model, prime_strokes, prime_text, line, char_to_idx, device,
                  tries=6, temp=0.35):
    best, best_score = None, float('inf')
    for _ in range(tries):
        pts, finished = sample_primed(model, prime_strokes, prime_text, line,
                                      char_to_idx, device, temp=temp)
        if not finished:
            continue                             

        # score is how far attempt is from median 9.6
        score = abs(len(pts) / len(line) - config.TARGET_SPC)
        if score < best_score:
            best, best_score = pts, score

    return best                  


