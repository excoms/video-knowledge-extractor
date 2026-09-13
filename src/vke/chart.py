"""A score-over-time chart, drawn with nothing but the standard library.

Deliberately dependency-free. matplotlib would be the obvious choice and it
would also add ~50MB to a download whose whole selling point is that it is
small enough to try. PNG is a simple enough container to write directly:
signature, IHDR, one zlib-compressed IDAT, IEND.

The chart answers one question — how did each speaker's standing move as the
debate went on, and where did it move for a reason worth naming.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

# A 3x5 bitmap font. Only the glyphs an axis needs: digits, colon, minus,
# plus and a space. Anything wordier belongs in the prose around the image.
FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    ":": ("000", "010", "000", "010", "000"),
    "-": ("000", "000", "111", "000", "000"),
    "+": ("000", "010", "111", "010", "000"),
    " ": ("000", "000", "000", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    "A": ("111", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "001"),
    "R": ("111", "101", "110", "101", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}


class Canvas:
    """An RGB pixel grid that knows how to save itself as a PNG."""

    def __init__(self, width: int, height: int, background=(255, 255, 255)):
        self.w, self.h = width, height
        self.px = bytearray(bytes(background) * (width * height))

    def set(self, x: int, y: int, colour) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.px[i:i + 3] = bytes(colour)

    def rect(self, x0: int, y0: int, x1: int, y1: int, colour) -> None:
        for y in range(max(0, y0), min(self.h, y1)):
            for x in range(max(0, x0), min(self.w, x1)):
                self.set(x, y, colour)

    def line(self, x0: int, y0: int, x1: int, y1: int, colour, width: int = 1) -> None:
        """Bresenham, thickened by drawing a small square at each step."""
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        half = width // 2
        while True:
            for oy in range(-half, half + 1):
                for ox in range(-half, half + 1):
                    self.set(x0 + ox, y0 + oy, colour)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def dashed(self, x0: int, x1: int, y: int, colour, on: int = 4, off: int = 4) -> None:
        x = x0
        while x < x1:
            self.rect(x, y, min(x + on, x1), y + 1, colour)
            x += on + off

    def disc(self, cx: int, cy: int, r: int, colour) -> None:
        for y in range(-r, r + 1):
            for x in range(-r, r + 1):
                if x * x + y * y <= r * r:
                    self.set(cx + x, cy + y, colour)

    def text(self, x: int, y: int, s: str, colour, scale: int = 1) -> int:
        """Draw with the 3x5 font. Returns the x the caret ended at."""
        for ch in s.upper():
            glyph = FONT.get(ch, FONT[" "])
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        self.rect(x + col * scale, y + row * scale,
                                  x + (col + 1) * scale, y + (row + 1) * scale, colour)
            x += 4 * scale
        return x

    def png(self, path: Path) -> Path:
        raw = bytearray()
        for y in range(self.h):
            raw.append(0)                      # filter type 0 for every scanline
            raw += self.px[y * self.w * 3:(y + 1) * self.w * 3]

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (struct.pack(">I", len(data)) + tag + data
                    + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

        ihdr = struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0)
        path.write_bytes(b"\x89PNG\r\n\x1a\n"
                         + chunk(b"IHDR", ihdr)
                         + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
                         + chunk(b"IEND", b""))
        return path


# The product palette, so a chart in a report looks like it belongs there.
INK = (20, 24, 28)
GRID = (216, 222, 220)
MUTED = (107, 118, 125)
SERIES = [(11, 122, 117), (180, 83, 9), (47, 125, 50), (108, 92, 160)]
BAD = (179, 38, 30)
GOOD = (47, 125, 50)


def _clock(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60}:{m % 60:02d}"


def momentum(series: dict, events: list[dict], path: Path,
             width: int = 900, height: int = 440) -> Path:
    """Cumulative score per speaker over the length of the debate.

    Two layers. Behind: one bar per scoring event, up for a point won, down
    for a fallacy or a shifted goalpost — so a bad ten minutes is visible as
    a cluster rather than inferred from a slope. In front: the running total,
    which is what actually decides the round.
    """
    speakers = [s for s in series if len(series[s]) > 1]
    if not speakers:
        raise ValueError("no speaker has any scoring events")

    L, R, T, B = 52, 18, 22, 34          # margins
    plot_w, plot_h = width - L - R, height - T - B

    duration = max((e["at"] for e in events), default=1) or 1
    highs = [v for s in speakers for _, v in series[s]]
    top, bottom = max(highs + [1]), min(highs + [-1])
    span = max(top - bottom, 1)
    # Leave headroom so the event bars never collide with the frame.
    top += span * 0.18
    bottom -= span * 0.18
    span = top - bottom

    def px(t: float) -> int:
        return L + int(plot_w * (t / duration))

    def py(v: float) -> int:
        return T + int(plot_h * (top - v) / span)

    c = Canvas(width, height, (255, 255, 255))
    zero = py(0)

    # Horizontal guides every 10 points, and the axis labels beside them.
    step = 10 if span > 40 else 5
    v = int(bottom // step) * step
    while v <= top:
        y = py(v)
        if T <= y < T + plot_h:
            c.rect(L, y, L + plot_w, y + 1, GRID)
            label = ("-" if v < 0 else "+" if v > 0 else " ") + str(abs(int(v)))
            c.text(6, y - 2, label, MUTED)
        v += step

    # Time axis, every ten minutes.
    for t in range(0, int(duration) + 1, 600):
        x = px(t)
        c.rect(x, T, x + 1, T + plot_h, GRID)
        c.text(max(L, x - 8), T + plot_h + 8, _clock(t), MUTED)

    # Zero is the line that matters; make it read differently from the grid.
    c.dashed(L, L + plot_w, zero, MUTED, on=5, off=4)

    # Layer one: the individual events, coloured by WHOSE they are.
    #
    # These were first drawn green for good and red for bad, which looked
    # sensible and answered the wrong question: a reader could see that a
    # fallacy happened but not who committed it. Direction already carries
    # the sign — up is a point won, down is a fallacy or a shifted goalpost —
    # so colour is free to carry attribution, which is the harder thing to
    # recover. Each speaker's bars also sit in their own narrow lane so two
    # events at the same moment do not overdraw each other.
    lane = {s: i for i, s in enumerate(speakers)}
    for e in events:
        if e["speaker"] not in speakers or not e["points"]:
            continue
        x = px(e["at"]) + lane[e["speaker"]] * 3
        h = abs(e["points"]) * 7
        colour = SERIES[lane[e["speaker"]] % len(SERIES)]
        if e["points"] > 0:
            c.rect(x, zero - h, x + 2, zero, colour)
        else:
            c.rect(x, zero + 1, x + 2, zero + 1 + h, colour)

    # Layer two: the running totals.
    for i, s in enumerate(speakers):
        colour = SERIES[i % len(SERIES)]
        pts = series[s]
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            c.line(px(t0), py(v0), px(t1), py(v0), colour, 2)   # hold
            c.line(px(t1), py(v0), px(t1), py(v1), colour, 2)   # step
        ex, ey = px(pts[-1][0]), py(pts[-1][1])
        c.disc(ex, ey, 4, colour)
        # Surname only: the font is 3px wide per glyph and a full name would
        # run off the plot. Stagger the row so two close lines stay readable.
        label = s.split()[-1][:12]
        total = pts[-1][1]
        lx = min(ex + 8, width - R - 4 * len(label) - 4 * len(str(total)) - 12)
        c.text(lx, ey - 10 if i % 2 == 0 else ey + 5, f"{label} {total:+d}", colour)

    return c.png(path)
