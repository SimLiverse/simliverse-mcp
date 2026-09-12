"""The transfer square at a tee pulls a carton to the centre, then turns it
(the user, 2026-09-12, on the photo of a carton riding the tee's near edge)."""

import numpy as np

from simliverse_sim.conveyor import Junction

C = (5.0, -1.0)
SIZE = (1.06, 0.9)
IN = (0.0, -1.0)     # the branch feeds from +y, cartons travel -y
OUT = (-1.0, 0.0)    # the line runs west


def test_nothing_over_the_square_waits_along_the_branch():
    assert Junction.direction_for(C, SIZE, IN, OUT, []) == "in"
    assert Junction.direction_for(C, SIZE, IN, OUT, [(5.0, 1.5, 0.87)]) == "in"    # still on the branch


def test_a_carton_short_of_the_centreline_is_pulled_in():
    # its centre 0.3 m before the line's centreline (y -0.7): keep pulling along -y
    assert Junction.direction_for(C, SIZE, IN, OUT, [(5.0, -0.7, 0.87)]) == "in"


def test_a_carton_on_the_centreline_is_sent_down_the_line():
    assert Junction.direction_for(C, SIZE, IN, OUT, [(5.0, -1.0, 0.87)]) == "out"
    assert Junction.direction_for(C, SIZE, IN, OUT, [(5.0, -1.1, 0.87)]) == "out"   # a touch past: still out
    assert Junction.direction_for(C, SIZE, IN, OUT, [(4.6, -1.0, 0.87)]) == "out"   # leaving along the line


def test_the_nearest_carton_decides():
    # one leaving down the line, one just arriving off the branch: the leaver is nearer the centre
    assert Junction.direction_for(C, SIZE, IN, OUT, [(4.7, -1.0, 0.87), (5.0, -0.3, 0.87)]) == "out"
    # once the leaver is off the square the arrival is pulled in
    assert Junction.direction_for(C, SIZE, IN, OUT, [(4.2, -1.0, 0.87), (5.0, -0.3, 0.87)]) == "in"


def test_the_junction_is_exported():
    import simliverse_sim

    assert simliverse_sim.Junction is Junction
