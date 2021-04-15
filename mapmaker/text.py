import os

from PIL import ImageFont

FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "fonts", "Cantarell-Regular.ttf"
)


def draw_text(draw, fontsize, chunks, x, y, align="left"):
    font = ImageFont.truetype(FONT_PATH, fontsize)
    widths = [draw.textsize(text, font)[0] for text, _ in chunks]
    y -= fontsize
    if isinstance(align, tuple):
        align, anchor = align
        x -= sum(widths[:anchor])
        anchorwidth = widths[anchor]
    else:
        anchorwidth = sum(widths)
    if align == "left":
        x += 0
    elif align == "right":
        x -= anchorwidth
    else:
        x -= anchorwidth // 2
    for w, (text, color) in zip(widths, chunks):
        draw.text(
            (x, y),
            text,
            font=font,
            fill=color,
        )
        x += w
