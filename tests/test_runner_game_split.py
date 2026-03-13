"""Test game allocation logic for parallel self-play."""

import pytest

from hybrid.rl.az_runner import _split_games_evenly


def test_split_games_evenly_30_4():
    alloc = _split_games_evenly(30, 4)
    assert alloc == [8, 8, 7, 7]
    assert sum(alloc) == 30
    assert max(alloc) - min(alloc) <= 1


def test_split_games_evenly_1_4():
    alloc = _split_games_evenly(1, 4)
    assert alloc == [1, 0, 0, 0]
    assert sum(alloc) == 1
    assert max(alloc) - min(alloc) <= 1


@pytest.mark.parametrize("total,workers", [(-1, 4), (10, 0), (10, -2)])
def test_split_games_evenly_invalid(total, workers):
    with pytest.raises(ValueError):
        _split_games_evenly(total, workers)
