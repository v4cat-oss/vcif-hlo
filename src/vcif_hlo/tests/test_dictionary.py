"""IdDictionary — interning round-trip; identity preservation."""
from __future__ import annotations

import pytest

from vcif_hlo import IdDictionary


def test_intern_assigns_increasing_ids():
    d = IdDictionary()
    assert d.intern('a') == 0
    assert d.intern('b') == 1
    assert d.intern('c') == 2


def test_intern_idempotent():
    d = IdDictionary()
    first = d.intern('alpha')
    assert d.intern('alpha') == first
    assert d.intern('alpha') == first  # twice more for good measure


def test_name_round_trip():
    d = IdDictionary()
    for s in ('alpha', 'beta', 'gamma'):
        d.intern(s)
    for i, s in enumerate(('alpha', 'beta', 'gamma')):
        assert d.name(i) == s


def test_len():
    d = IdDictionary()
    assert len(d) == 0
    d.intern('a')
    assert len(d) == 1
    d.intern('a')
    assert len(d) == 1
    d.intern('b')
    assert len(d) == 2


def test_contains_only_strings():
    d = IdDictionary()
    d.intern('alpha')
    assert 'alpha' in d
    assert 'beta' not in d
    assert 0 not in d
    assert None not in d


def test_intern_rejects_non_strings():
    d = IdDictionary()
    with pytest.raises(TypeError):
        d.intern(42)


def test_intern_many():
    d = IdDictionary()
    ids = d.intern_many(['a', 'b', 'a', 'c'])
    assert ids == [0, 1, 0, 2]


def test_iter_returns_pairs_in_insertion_order():
    d = IdDictionary()
    for s in ('z', 'y', 'x'):
        d.intern(s)
    assert list(d) == [(0, 'z'), (1, 'y'), (2, 'x')]


def test_get_default_for_missing():
    d = IdDictionary()
    assert d.get('missing') is None
    assert d.get('missing', -1) == -1
    d.intern('present')
    assert d.get('present') == 0
