#!/usr/bin/env python3
"""Generate the RivenQuant source logo (1024x1024 PNG) for `tauri icon`."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw

SIZE = 1024
OUT = os.path.join(os.path.dirname(__file__), "logo.png")


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def main() -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Rounded-square dark background with a subtle vertical gradient.
    top, bottom = (22, 22, 30), (12, 12, 16)
    radius = int(SIZE * 0.22)
    bg = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    bgd = ImageDraw.Draw(bg)
    for y in range(SIZE):
        bgd.line([(0, y), (SIZE, y)], fill=_lerp(top, bottom, y / SIZE) + (255,))
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=radius, fill=255)
    img.paste(bg, (0, 0), mask)

    # Ascending gradient "quant" line with nodes (indigo -> cyan).
    indigo, cyan = (99, 102, 241), (34, 211, 238)
    pts = [
        (0.20, 0.74), (0.37, 0.60), (0.50, 0.66),
        (0.64, 0.42), (0.80, 0.30),
    ]
    px = [(int(x * SIZE), int(y * SIZE)) for x, y in pts]
    width = int(SIZE * 0.045)
    for i in range(len(px) - 1):
        t = i / (len(px) - 2)
        d.line([px[i], px[i + 1]], fill=_lerp(indigo, cyan, t) + (255,), width=width, joint="curve")
    for i, (x, y) in enumerate(px):
        t = i / (len(px) - 1)
        r = int(SIZE * 0.032)
        d.ellipse([x - r, y - r, x + r, y + r], fill=_lerp(indigo, cyan, t) + (255,))

    # Arrow head at the last node.
    hx, hy = px[-1]
    a = int(SIZE * 0.055)
    d.polygon([(hx + a, hy - a), (hx + a, hy + int(a * 0.2)), (hx - int(a * 0.2), hy - a)],
              fill=cyan + (255,))

    img.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
