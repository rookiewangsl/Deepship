from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path("/home/shilongwang/Transformer/image_redrawn.png")
W, H = 2100, 1700


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


REG = load_font(26)
SMALL = load_font(22)
MINI = load_font(18)
BOLD = load_font(30, True)
BOLD_SMALL = load_font(24, True)

BLUE = "#5f7fcb"
BLUE_EDGE = "#2f4670"
GRAY = "#d9d9d9"
YELLOW = "#f6ddb0"
CYAN = "#bfe7f8"
PURPLE = "#5d72b9"
RED = "#cc6f6f"
BLACK = "#2b2b2b"
FRAME = "#90b2e8"


img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def center_text(box, text, font, fill=BLACK, line_gap=6):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    sizes = [d.textbbox((0, 0), line, font=font) for line in lines]
    widths = [b[2] - b[0] for b in sizes]
    heights = [b[3] - b[1] for b in sizes]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, w, h in zip(lines, widths, heights, strict=True):
        x = x1 + (x2 - x1 - w) / 2
        d.text((x, y), line, font=font, fill=fill)
        y += h + line_gap


def rounded(box, fill, outline=BLACK, width=3, radius=8):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def stacked_box(x, y, w, h, text, fill=BLUE, outline=BLUE_EDGE, font=SMALL, text_fill="white"):
    for dx, dy in [(10, -10), (5, -5), (0, 0)]:
        rounded((x + dx, y + dy, x + w + dx, y + h + dy), fill, outline, width=3, radius=4)
    center_text((x, y, x + w, y + h), text, font, fill=text_fill)


def arrow(p1, p2, fill=BLUE_EDGE, width=4):
    d.line([p1, p2], fill=fill, width=width)
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) >= abs(y2 - y1):
        s = 1 if x2 > x1 else -1
        pts = [(x2, y2), (x2 - 16 * s, y2 - 8), (x2 - 16 * s, y2 + 8)]
    else:
        s = 1 if y2 > y1 else -1
        pts = [(x2, y2), (x2 - 8, y2 - 16 * s), (x2 + 8, y2 - 16 * s)]
    d.polygon(pts, fill=fill)


def dashed_segment(p1, p2, fill=RED, width=3, dash=10, gap=7):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        s = 1 if y2 > y1 else -1
        step = dash + gap
        for y in range(y1, y2, step * s):
            d.line([(x1, y), (x1, y + dash * s)], fill=fill, width=width)
    elif y1 == y2:
        s = 1 if x2 > x1 else -1
        step = dash + gap
        for x in range(x1, x2, step * s):
            d.line([(x, y1), (x + dash * s, y1)], fill=fill, width=width)
    else:
        d.line([p1, p2], fill=fill, width=width)


def dashed_polyline(points, fill=RED, width=3):
    for a, b in zip(points, points[1:]):
        dashed_segment(a, b, fill=fill, width=width)
    arrow(points[-2], points[-1], fill=fill, width=width)


d.text((760, 35), "Three-Branch MA-CNN-A Structural Diagram", font=BOLD, fill=BLACK)

# Input
rounded((910, 60, 1190, 120), "white", outline=BLUE_EDGE, width=3, radius=4)
center_text((910, 60, 1190, 120), "Input", BOLD_SMALL)

branch_x = [170, 810, 1450]
labels1 = ["1×8 32", "1×16 32", "1×32 32"]
labels2 = ["8×1 32", "16×1 32", "32×1 32"]
labels3 = ["1×8 64", "1×16 64", "1×32 64"]
labels4 = ["8×1 64", "16×1 64", "32×1 64"]
bottom_centers = []
upper_weight_centers = []
lower_weight_centers = []

rounded((90, 360, 1950, 1110), None, outline=FRAME, width=4, radius=28)

for i, x in enumerate(branch_x):
    stacked_box(x, 170, 150, 46, labels1[i])
    arrow((1050, 120), (x + 75, 170))
    stacked_box(x, 285, 150, 46, labels2[i])
    arrow((x + 75, 216), (x + 75, 285))
    stacked_box(x, 500, 170, 52, labels3[i])
    arrow((x + 75, 331), (x + 85, 500))
    stacked_box(x, 715, 170, 52, labels4[i])
    arrow((x + 85, 552), (x + 85, 715))
    bottom_centers.append((x + 85, 767))

    # upper attention path
    rounded((x + 210, 545, x + 315, 580), GRAY, outline=BLACK, width=2, radius=2)
    rounded((x + 210, 605, x + 315, 662), YELLOW, outline=BLACK, width=2, radius=2)
    center_text((x + 210, 605, x + 315, 662), "k", BOLD_SMALL)
    dashed_polyline([(x + 85, 552), (x + 262, 545)])
    ux1, uy1 = x + 185, 695
    for j, c in enumerate(["#ff8c00", "#2ca25f", "#000000", "#6baed6", "#fdd0a2", "#3182bd", "#e34a33", "#756bb1"]):
        d.rectangle((ux1 + j * 18, uy1, ux1 + (j + 1) * 18, uy1 + 34), fill=c)
    upper_weight_centers.append((ux1 + 72, uy1 + 17))

    # lower attention path
    rounded((x + 210, 790, x + 315, 825), GRAY, outline=BLACK, width=2, radius=2)
    rounded((x + 210, 850, x + 315, 907), YELLOW, outline=BLACK, width=2, radius=2)
    center_text((x + 210, 850, x + 315, 907), "k", BOLD_SMALL)
    dashed_polyline([(x + 85, 767), (x + 262, 790)])
    lx1, ly1 = x + 185, 940
    for j, c in enumerate(["#b2182b", "#fdae61", "#000000", "#67a9cf", "#2166ac", "#fdb863", "#1a9850", "#d73027"]):
        d.rectangle((lx1 + j * 18, ly1, lx1 + (j + 1) * 18, ly1 + 34), fill=c)
    lower_weight_centers.append((lx1 + 72, ly1 + 17))

# connect weight strips horizontally
for left, right in zip(upper_weight_centers, upper_weight_centers[1:]):
    dashed_polyline([(left[0] + 74, left[1]), (right[0] - 74, right[1])])
for left, right in zip(lower_weight_centers, lower_weight_centers[1:]):
    dashed_polyline([(left[0] + 74, left[1]), (right[0] - 74, right[1])])

# feature add
add_center = (980, 980)
for c in bottom_centers:
    arrow(c, add_center)
d.ellipse((add_center[0] - 35, add_center[1] - 35, add_center[0] + 35, add_center[1] + 35), fill=PURPLE, outline=BLUE_EDGE, width=3)
center_text((add_center[0] - 35, add_center[1] - 35, add_center[0] + 35, add_center[1] + 35), "+", BOLD, fill="white")
rounded((915, 1030, 1045, 1080), "#e6e6e6", outline="#999999", width=2, radius=2)

# weight add
w_add = (1750, 880)
d.ellipse((w_add[0] - 35, w_add[1] - 35, w_add[0] + 35, w_add[1] + 35), fill=PURPLE, outline=BLUE_EDGE, width=3)
center_text((w_add[0] - 35, w_add[1] - 35, w_add[0] + 35, w_add[1] + 35), "+", BOLD, fill="white")
dashed_polyline([(upper_weight_centers[-1][0] + 74, upper_weight_centers[-1][1]), (1715, 868)])
dashed_polyline([(lower_weight_centers[-1][0] + 74, lower_weight_centers[-1][1]), (1715, 892)])
wx1, wy1 = 1810, 862
weight_cols = ["#ff0000", "#ff8c00", "#ffff00", "#7fc97f", "#c0c0c0", "#90ee90", "#9ecae1", "#756bb1"]
for j, c in enumerate(weight_cols):
    d.rectangle((wx1 + j * 24, wy1, wx1 + (j + 1) * 24, wy1 + 36), fill=c)

# multiply
mul_center = (1125, 1088)
dashed_polyline([(1940, 898), (1940, 1090), (1160, 1090)])
d.ellipse((mul_center[0] - 35, mul_center[1] - 35, mul_center[0] + 35, mul_center[1] + 35), fill=PURPLE, outline=BLUE_EDGE, width=3)
center_text((mul_center[0] - 35, mul_center[1] - 35, mul_center[0] + 35, mul_center[1] + 35), "*", BOLD, fill="white")
arrow((1045, 1055), (1090, 1088))

# weighted feature map
fx1, fy1 = 1190, 1048
for j, c in enumerate(["#e41a1c", "#ff7f00", "#ffff33", "#a6cee3", "#1f78b4", "#b2df8a", "#cab2d6", "#6a3d9a"]):
    d.polygon([(fx1 + j * 13, fy1), (fx1 + 18 + j * 13, fy1 - 10), (fx1 + 18 + j * 13, fy1 + 72), (fx1 + j * 13, fy1 + 82)], fill=c)

# final head, with more spacing and complete boxes
stacked_box(1120, 1185, 165, 54, "1×8 98")
arrow((1240, 1130), (1205, 1185))
stacked_box(1120, 1275, 165, 54, "8×1 98")
arrow((1205, 1239), (1205, 1275))
rounded((1080, 1370, 1330, 1420), GRAY, outline=BLACK, width=2, radius=2)
center_text((1080, 1370, 1330, 1420), "GAP", BOLD_SMALL)
arrow((1205, 1329), (1205, 1370))
rounded((1040, 1450, 1370, 1545), CYAN, outline=BLACK, width=2, radius=3)
center_text((1040, 1450, 1370, 1545), "FC\nsoftmax", BOLD_SMALL)
arrow((1205, 1420), (1205, 1450))

# left CBA legend
d.rounded_rectangle((110, 1180, 470, 1600), radius=26, outline=FRAME, width=4)
d.text((160, 1210), "CBA block", font=BOLD_SMALL, fill=BLACK)
d.text((280, 1260), "input", font=MINI, fill=BLACK)
arrow((300, 1286), (300, 1315), fill=BLACK, width=3)
rounded((170, 1315, 430, 1415), "white", outline=BLACK, width=2, radius=20)
center_text((170, 1315, 430, 1415), "Conv2D\n(kernel size,\nfilters)", SMALL)
rounded((170, 1440, 430, 1500), "white", outline=BLACK, width=2, radius=16)
center_text((170, 1440, 430, 1500), "BN", SMALL)
rounded((170, 1525, 430, 1585), "white", outline=BLACK, width=2, radius=16)
center_text((170, 1525, 430, 1585), "Activation", SMALL)
d.text((270, 1595), "output", font=MINI, fill=BLACK)

# right legend
lx, ly = 1570, 1200
legend = [
    ("Conv2D, BN, Activation  (CBA)", "blue"),
    ("GAP", "gray"),
    ("Full connection-Softmax", "cyan"),
    ("Weight Normalization-Sigmoid", "strip"),
    ("Conv1D", "yellow"),
    ("Multiply", "mul"),
    ("Add", "add"),
]
for i, (name, kind) in enumerate(legend):
    y = ly + i * 52
    if kind == "blue":
        rounded((lx, y, lx + 60, y + 30), BLUE, outline=BLUE_EDGE, width=2, radius=4)
    elif kind == "gray":
        rounded((lx, y, lx + 60, y + 30), GRAY, outline=BLACK, width=2, radius=2)
    elif kind == "cyan":
        rounded((lx, y, lx + 60, y + 30), CYAN, outline=BLACK, width=2, radius=2)
    elif kind == "yellow":
        rounded((lx, y, lx + 60, y + 30), YELLOW, outline=BLACK, width=2, radius=2)
    elif kind == "strip":
        for j, c in enumerate(weight_cols):
            d.rectangle((lx + j * 7, y, lx + (j + 1) * 7, y + 30), fill=c)
    elif kind == "mul":
        d.ellipse((lx, y, lx + 30, y + 30), fill=PURPLE, outline=BLUE_EDGE, width=2)
        center_text((lx, y, lx + 30, y + 30), "*", BOLD_SMALL, fill="white")
    elif kind == "add":
        d.ellipse((lx, y, lx + 30, y + 30), fill=PURPLE, outline=BLUE_EDGE, width=2)
        center_text((lx, y, lx + 30, y + 30), "+", BOLD_SMALL, fill="white")
    d.text((lx + 85, y - 1), name, font=SMALL, fill=BLACK)

d.text((680, 1630), "Figure 8. Re-drawn structural diagram of the three-branch MA-CNN-A model.", font=BOLD_SMALL, fill=BLACK)

img.save(OUT)
print(OUT)
