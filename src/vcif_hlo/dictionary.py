"""
vcif_hlo.dictionary — the string ↔ Id boundary.

Per the writeup § 2: OpenHLO never reasons over strings. The
dictionary is intentionally OUTSIDE the semantic core. It exists at
the import boundary; once a string is interned, Id tensors flow
through the kernels without any string lookups.

Stable assignment: IDs increment from 0; once an identifier is
interned, its ID is permanent for the lifetime of the dictionary.
"""
from __future__ import annotations

from typing import Iterable


class IdDictionary:
    """Bidirectional mapping between string identifiers and integer Ids.

    Stable: once interned, an Id is permanent for the dictionary's lifetime.

    Example
    -------
    >>> d = IdDictionary()
    >>> d.intern("alpha")
    0
    >>> d.intern("beta")
    1
    >>> d.intern("alpha")          # idempotent
    0
    >>> d.name(1)
    'beta'
    >>> len(d)
    2
    """

    __slots__ = ('_to_id', '_to_name')

    def __init__(self):
        self._to_id: dict[str, int] = {}
        self._to_name: list[str] = []

    def intern(self, identifier: str) -> int:
        """Return the Id for `identifier`, allocating a fresh one if new."""
        if not isinstance(identifier, str):
            raise TypeError(f'identifier must be str, got {type(identifier).__name__}')
        existing = self._to_id.get(identifier)
        if existing is not None:
            return existing
        new_id = len(self._to_name)
        self._to_id[identifier] = new_id
        self._to_name.append(identifier)
        return new_id

    def intern_many(self, identifiers: Iterable[str]) -> list[int]:
        """Intern a sequence of identifiers; return Ids in order."""
        return [self.intern(s) for s in identifiers]

    def name(self, id_: int) -> str:
        """Return the string identifier for `id_`. Raises IndexError if unknown."""
        return self._to_name[id_]

    def has(self, identifier: str) -> bool:
        return identifier in self._to_id

    def has_id(self, id_: int) -> bool:
        return 0 <= id_ < len(self._to_name)

    def get(self, identifier: str, default: int | None = None) -> int | None:
        """Return the Id for `identifier`, or `default` if not yet interned."""
        return self._to_id.get(identifier, default)

    def __len__(self) -> int:
        return len(self._to_name)

    def __contains__(self, identifier: object) -> bool:
        return isinstance(identifier, str) and identifier in self._to_id

    def __iter__(self):
        """Iterate (id, name) pairs in insertion order."""
        return enumerate(self._to_name)

    def __repr__(self) -> str:
        return f'IdDictionary(size={len(self)})'

    def names(self) -> list[str]:
        """All interned strings in Id order. Reads, doesn't copy."""
        return list(self._to_name)
