import typing as t


class Vocab:
    def __init__(
        self,
        tokens: t.Iterable[str],
        specials: t.Optional[t.Iterable[str]] = None,
        default_index: t.Optional[int] = None,
    ):
        self._stoi = {}
        self._itos = []
        if specials is None:
            specials = []
        i = -1
        for i, token in enumerate(specials):
            self._itos.append(token)
            self._stoi[token] = i
        for i, token in enumerate(tokens, start=i + 1):
            self._itos.append(token)
            self._stoi[token] = i
        self._default_index = default_index

    def set_default_index(self, i):
        self._default_index = i

    @property
    def default_index(self):
        if self._default_index is None:
            raise ValueError("default index undefined")
        return self._default_index

    @property
    def stoi(self):
        return self._stoi

    @property
    def itos(self):
        return self._itos

    def __len__(self):
        return len(self._itos)

    def __getitem__(self, key):
        out = self._stoi.get(key)
        if out is None:
            return self.default_index
        return out

    def lookup_indices(self, tokens):
        return [self.stoi[token] for token in tokens]

    def lookup_tokens(self, ids):
        return [self.itos[x] for x in ids]
