import csv
import typing as t


class CSVChunkWriter:
    def __init__(self, path_format_str, header, n_lines_per_chunk=50000):
        self._fmt_str = path_format_str
        self._header = header
        self._line_count = 0
        self._chunk_count = 0
        self._writer = None
        self._modulo = n_lines_per_chunk - 1
        self._outf = None

    def writerow(self, row: t.Iterable[str]):
        if self._writer is None or (not self._line_count % self._modulo):
            if self._outf is not None:
                self._outf.close()
            self._chunk_count += 1
            path = self._fmt_str.format(self._chunk_count)
            self._outf = open(path, "w", newline="")
            self._writer = csv.writer(self._outf, delimiter=",")
            self._writer.writerow(self._header)
        self._writer.writerow(row)
        self._line_count += 1

    def close(self):
        if self._outf is not None:
            self._outf.close()

    def write(self, *elements):
        self.writerow(
            (
                " ".join(item) if isinstance(item, t.Sequence) else item
                for item in elements
            )
        )
