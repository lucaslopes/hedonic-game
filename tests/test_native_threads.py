"""Detections and audits from several threads at once.

Every result must be the one the same call gives alone, bit for bit, and
each caller's random.Random must end where its own serial call leaves it.
The Rust core runs the visits and the audit's queries without the GIL, so
the calls really overlap there (the C core holds the GIL: they take turns,
with the same results); `test_gil_released` checks that on the Rust core.
"""

import random
import threading
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

import pytest

igraph = pytest.importorskip("igraph")
np = pytest.importorskip("numpy")

from hedonic import Game, _core  # noqa: E402

THREADS = 8


def bits(x):
    """Floats by their bits, recursively (NaN equal to NaN)."""
    if isinstance(x, float):
        return x.hex()
    if isinstance(x, (list, tuple)):
        return [bits(v) for v in x]
    if isinstance(x, np.ndarray):
        return [v.hex() for v in x.ravel().tolist()]
    return x


def graphs():
    rnd = random.Random(20261020)
    out = [("zachary", igraph.Graph.Famous("Zachary"), None)]
    random.seed(1)
    ba = igraph.Graph.Barabasi(3000, 3)
    out.append(("ba3000", ba, None))
    blocks = [[0.08 if i == j else 0.004 for j in range(6)] for i in range(6)]
    random.seed(2)
    sbm = igraph.Graph.SBM(blocks, [150] * 6)
    out.append(("sbm900 weighted", sbm, array("d", [rnd.uniform(0.2, 2.0) for _ in range(sbm.ecount())])))
    return out


def calls():
    """(name, thunk) pairs: each thunk makes its own generator and returns
    the result with the generator's state after the call."""
    out = []
    for name, g, w in graphs():
        density = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
        for cap in (1, 3):
            for seed in (0, 7):
                def detect(g=g, w=w, cap=cap, seed=seed, gamma=density * (1 if cap == 1 else 3)):
                    r = random.Random(seed)
                    return _core.detect(g, r, gamma, max_memberships=cap, edge_weights=w), r.getstate()
                out.append((f"{name} detect cap {cap} seed {seed}", detect))

        def plain(g=g, seed=3, gamma=density):
            flat = array("q", [x for e in g.get_edgelist() for x in e])
            r = random.Random(seed)
            return _core.partition(g.vcount(), flat, r, resolution=gamma), r.getstate()
        out.append((f"{name} partition()", plain))

        cover = Game(g).community_hedonic(max_memberships=3, resolution=3 * density, seed=11)
        rows = [list(r) for r in cover.membership]
        grid = np.linspace(0.0, 1.0, 9)

        def intervals(g=g, rows=rows):
            return Game(g).stable_intervals(rows, max_memberships=3)

        def regrets(g=g, rows=rows, grid=grid):
            return tuple(Game(g).regrets(grid, rows, max_memberships=3, allow_isolation=True))
        out.append((f"{name} stable_intervals", intervals))
        out.append((f"{name} regrets", regrets))
    return out


def test_concurrent_calls_match_serial():
    todo = calls()
    serial = {name: bits(thunk()) for name, thunk in todo}
    work = [item for _ in range(3) for item in todo]
    random.Random(5).shuffle(work)
    with ThreadPoolExecutor(THREADS) as pool:
        got = list(pool.map(lambda item: (item[0], bits(item[1]())), work))
    for name, result in got:
        assert result == serial[name], name


def test_gil_released():
    """While a long detection runs on another thread, this thread keeps
    running Python code (it could not while the C core holds the GIL)."""
    if getattr(_core, "backend", "c") != "rust":
        pytest.skip("the C core holds the GIL during a call")
    random.seed(1)
    g = igraph.Graph.Barabasi(60000, 4)
    gamma = 4 * 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
    window = {}
    r = random.Random(1)

    def worker():
        window["start"] = time.perf_counter()
        _core.detect(g, r, gamma, max_memberships=3)
        window["end"] = time.perf_counter()

    stamps = []
    thread = threading.Thread(target=worker)
    thread.start()
    k = 0
    while thread.is_alive():
        k += 1
        if k % 1000 == 0:
            stamps.append(time.perf_counter())
    thread.join()
    start, end = window["start"], window["end"]
    # The middle half of the call: the argument checks before it and the
    # result's lists after it run with the GIL (the C core lets this thread
    # in only there, past 95% of the call), the visits in between without.
    middle = [t for t in stamps if start + 0.25 * (end - start) < t < start + 0.75 * (end - start)]
    assert len(middle) > 10, (len(middle), end - start)


def test_interrupt_from_main_thread():
    """A signal handler that raises stops a detection on the main thread,
    with the GIL released, and the generator's state is still written back."""
    import signal

    if not hasattr(signal, "setitimer"):
        pytest.skip("no setitimer")
    random.seed(1)
    g = igraph.Graph.Barabasi(200000, 5)
    gamma = 4 * 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))

    class Stop(Exception):
        pass

    def handler(signum, frame):
        raise Stop()

    old = signal.signal(signal.SIGALRM, handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.05)
        r = random.Random(0)
        with pytest.raises(Stop):
            _core.detect(g, r, gamma, max_memberships=4)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    r.random()  # still a working generator
