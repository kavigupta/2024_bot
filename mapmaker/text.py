import matplotlib.font_manager

from PIL import ImageFont

[FONT_PATH] = [
    x
    for x in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    if "Cantarell-Regular" in x
]


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
