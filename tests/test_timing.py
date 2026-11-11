"""Chronométrage de la calibration : cumul par étape, appels, et découpage (Lap)."""
import threading

from astrostack.timing import Chrono, Lap, measure


def test_chrono_accumulates_per_step_and_counts_calls():
    c = Chrono()
    c.add("lecture", 0.5)
    c.add("lecture", 0.25)
    c.add("dématriçage", 2.0)
    s = c.summary()
    assert "lecture 0.75 s (2×" in s
    assert "dématriçage 2.00 s (1×" in s
    assert s.index("lecture") < s.index("dématriçage")  # ordre d'apparition


def test_measure_context_records_elapsed_time():
    c = Chrono()
    with c.measure("étape"):
        pass
    assert "étape" in c.summary()


def test_lap_splits_consecutive_steps():
    c = Chrono()
    lap = Lap(c)
    lap.mark("a")
    lap.mark("b")
    assert "a " in c.summary() and "b " in c.summary()


def test_disabled_chrono_is_a_no_op():
    lap = Lap(None)
    lap.mark("x")  # ne doit rien lever
    with measure(None, "y"):
        pass


def test_chrono_is_thread_safe():
    c = Chrono()

    def work():
        for _ in range(500):
            c.add("t", 0.001)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert "t 2.00 s (2000×" in c.summary()
