from .hedonics import hedonic
import random

def stochastic(c):
    here = random.choice(c)
    while here == {}:
        here = random.choice(c)
    there = random.choice(c)
    while there == here:
        there = random.choice(c)
    node = random.choice(list(here))
    chosen = [node, c.index(here), c.index(there)]
    return chosen

def partial(c):
    here = random.choice(c)
    while here == {}:
        here = random.choice(c)
    node = random.choice(list(here))
    chosen = [None, None, None, 0]
    current = hedonic(node, here)
    for there in c:
        increased = hedonic(node, there) - current
        if increased > chosen[3]:
            chosen = [node, c.index(here), c.index(there), increased]
    return chosen[:3]

def greedy(c):
    chosen = [None, None, None, 0]
    for here in c:
        for key in here:
            current = hedonic(key, here)
            for there in c:
                increased = hedonic(key, there) - current
                if increased > chosen[3]:
                    chosen = [key, c.index(here), c.index(there), increased]
    return chosen[:3]

def doNothing(c):
    return None
