"""P1 curation test (CPU-only): random record split. No pipeline run."""

from data_pipeline.split import split_records


def _recs(n):
    return [(f">acc{i}_IDR_1-30", "M" * 30) for i in range(n)]


def test_split_sizes_and_partition():
    recs = _recs(1000)
    train, val, test = split_records(recs, (0.99, 0.005, 0.005), seed=0)
    assert (len(train), len(val), len(test)) == (990, 5, 5)
    # disjoint and exhaustive
    allr = train + val + test
    assert len(allr) == 1000
    assert {h for h, _ in allr} == {h for h, _ in recs}


def test_split_is_deterministic_by_seed():
    recs = _recs(100)
    assert split_records(recs, seed=7)[0] == split_records(recs, seed=7)[0]
    assert split_records(recs, seed=7)[0] != split_records(recs, seed=8)[0]
