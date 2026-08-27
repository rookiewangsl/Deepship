from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


NAVY = "#17324D"
BLUE = "#2D6FA3"
BLUE_FILL = "#EAF3F8"
TEAL = "#268A88"
TEAL_FILL = "#E5F4F1"
ORANGE = "#D97735"
ORANGE_FILL = "#FAEEE5"
PURPLE = "#675A9C"
PURPLE_FILL = "#EEEAF7"
GRAY = "#637381"
LINE = "#9AAAB6"
LIGHT_LINE = "#D8E0E5"
PANEL = "#F8FAFB"
WHITE = "#FFFFFF"


def setup_axes(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.axis("off")
    return fig, ax


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = LINE,
    linewidth: float = 1.2,
    radius: float = 0.8,
    linestyle: str = "-",
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text(
    ax,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 10,
    color: str = NAVY,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    zorder: int = 6,
):
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontfamily="DejaVu Sans",
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=1.22,
        zorder=zorder,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    linewidth: float = 1.5,
    linestyle: str = "-",
    connectionstyle: str = "arc3",
    head: bool = True,
    zorder: int = 4,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>" if head else "-",
        mutation_scale=11,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def operator(ax, x: float, y: float, symbol: str, *, color: str):
    patch = Circle((x, y), 1.8, facecolor=color, edgecolor=WHITE, linewidth=1.5, zorder=6)
    ax.add_patch(patch)
    text(ax, x, y - 0.05, symbol, size=13, color=WHITE, weight="bold", zorder=7)


def panel_label(ax, x: float, y: float, label: str, title: str):
    text(ax, x, y, f"{label}  {title}", size=12, color=NAVY, weight="bold", ha="left")


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=240 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor=WHITE,
        )
    plt.close(fig)


def draw_spectrogram_icon(ax, x: float, y: float, width: float, height: float):
    colors = ["#EAF3F8", "#B9DAE8", "#72B4C5", "#F1C27D", "#D97735"]
    rows = 5
    cols = 9
    cell_w = width / cols
    cell_h = height / rows
    values = [
        [0, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 2, 1, 1, 2, 1, 0, 1],
        [1, 2, 3, 2, 3, 2, 1, 2, 1],
        [2, 3, 4, 3, 4, 3, 2, 3, 2],
        [3, 4, 4, 4, 4, 4, 3, 4, 3],
    ]
    for row in range(rows):
        for col in range(cols):
            box(
                ax,
                x + col * cell_w,
                y + (rows - row - 1) * cell_h,
                cell_w + 0.02,
                cell_h + 0.02,
                face=colors[values[row][col]],
                edge=colors[values[row][col]],
                linewidth=0,
                radius=0.0,
                zorder=4,
            )


def branch_summary(ax, x: float, y: float, kernel: int):
    width = 37.0
    height = 9.0
    box(ax, x, y, width, height, face=BLUE_FILL, edge=BLUE, linewidth=1.4, radius=1.0)
    box(ax, x + 1.0, y + 1.1, 5.1, height - 2.2, face=BLUE, edge=BLUE, radius=0.7)
    text(ax, x + 3.55, y + height / 2, f"k={kernel}", size=9.5, color=WHITE, weight="bold")
    layers = [("1 x k", "32"), ("k x 1", "32"), ("1 x k", "64"), ("k x 1", "64")]
    start_x = x + 7.4
    layer_w = 6.35
    gap = 0.7
    for index, (kernel_label, channels) in enumerate(layers):
        layer_x = start_x + index * (layer_w + gap)
        face = WHITE if index < 2 else "#D9EAF3"
        box(ax, layer_x, y + 1.1, layer_w, height - 2.2, face=face, edge=LIGHT_LINE, linewidth=0.9, radius=0.55)
        text(ax, layer_x + layer_w / 2, y + 5.5, kernel_label.replace("k", str(kernel)), size=8.0, color=NAVY, weight="bold")
        text(ax, layer_x + layer_w / 2, y + 2.8, f"C={channels}", size=7.5, color=GRAY)
        if index < 3:
            arrow(
                ax,
                (layer_x + layer_w + 0.05, y + height / 2),
                (layer_x + layer_w + gap - 0.05, y + height / 2),
                color=LINE,
                linewidth=1.0,
            )


def draw_main_figure(output_dir: Path):
    fig, ax = setup_axes((18.5, 9.4))
    ax.set_xlim(0, 184)
    ax.set_ylim(0, 94)

    panel_label(ax, 2.0, 90.5, "(a)", "Multi-scale asymmetric backbone")

    box(ax, 2.0, 36.0, 20.0, 43.0, face=PANEL, edge=LIGHT_LINE, linewidth=1.1, radius=1.4)
    text(ax, 12.0, 75.0, "Log-Mel input", size=11.0, color=NAVY, weight="bold")
    draw_spectrogram_icon(ax, 5.0, 58.5, 14.0, 11.0)
    arrow(ax, (12.0, 57.0), (12.0, 51.7), color=BLUE, linewidth=1.5)
    text(ax, 12.0, 48.4, "B x 1 x 64 x 94", size=10.0, color=BLUE, weight="bold")
    text(ax, 12.0, 42.0, "Frequency", size=8.0, color=GRAY)
    arrow(ax, (7.0, 39.5), (17.0, 39.5), color=ORANGE, linewidth=1.2)
    text(ax, 12.0, 37.2, "Time", size=8.0, color=ORANGE)

    branch_y = [67.0, 53.0, 39.0]
    kernels = [8, 16, 32]
    for y, kernel in zip(branch_y, kernels, strict=True):
        branch_summary(ax, 29.0, y, kernel)
        arrow(
            ax,
            (22.4, 49.0),
            (28.6, y + 4.5),
            color=LINE,
            linewidth=1.2,
            connectionstyle="arc3,rad=0.08",
        )
        arrow(
            ax,
            (66.4, y + 4.5),
            (73.6, 58.0),
            color=BLUE,
            linewidth=1.3,
            connectionstyle="arc3,rad=0.07",
        )

    text(ax, 47.5, 82.7, "Parallel receptive fields", size=9.0, color=GRAY, weight="bold")
    operator(ax, 76.0, 58.0, "+", color=BLUE)
    text(ax, 76.0, 63.0, "Element-wise sum", size=8.2, color=BLUE, weight="bold")

    box(ax, 81.0, 52.3, 18.0, 11.4, face=BLUE_FILL, edge=BLUE, linewidth=1.4, radius=1.0)
    text(ax, 90.0, 60.2, "Fused feature", size=9.5, color=BLUE, weight="bold")
    text(ax, 90.0, 55.8, "B x 64 x 34 x 49", size=9.2, color=NAVY, weight="bold")
    arrow(ax, (78.0, 58.0), (80.6, 58.0), color=BLUE, linewidth=1.4)

    panel_label(ax, 72.0, 31.5, "(b)", "Six-source ECA attention")
    box(ax, 72.0, 6.0, 58.0, 20.5, face=PANEL, edge=LIGHT_LINE, linewidth=1.1, radius=1.4)
    box(ax, 75.0, 11.0, 13.0, 10.5, face=WHITE, edge=TEAL, linewidth=1.2, radius=0.8)
    text(ax, 81.5, 18.3, "6 feature maps", size=8.7, color=TEAL, weight="bold")
    text(ax, 81.5, 14.3, "L3 + L4 from\n3 branches", size=8.0, color=GRAY)
    arrow(ax, (88.4, 16.2), (91.0, 16.2), color=TEAL, linewidth=1.3)

    box(ax, 91.4, 11.0, 9.0, 10.5, face=TEAL_FILL, edge=TEAL, linewidth=1.2, radius=0.8)
    text(ax, 95.9, 17.9, "GAP", size=9.2, color=TEAL, weight="bold")
    text(ax, 95.9, 14.1, "C=64", size=8.0, color=GRAY)
    arrow(ax, (100.8, 16.2), (103.1, 16.2), color=TEAL, linewidth=1.3)

    box(ax, 103.5, 11.0, 12.0, 10.5, face=TEAL_FILL, edge=TEAL, linewidth=1.2, radius=0.8)
    text(ax, 109.5, 18.0, "Conv1D", size=9.0, color=TEAL, weight="bold")
    text(ax, 109.5, 14.1, "k=3 + Sigmoid", size=8.0, color=GRAY)
    arrow(ax, (115.9, 16.2), (118.0, 16.2), color=TEAL, linewidth=1.3)
    operator(ax, 120.0, 16.2, "+", color=TEAL)
    text(ax, 120.0, 22.9, "Sum 6 weights", size=8.0, color=TEAL, weight="bold")
    arrow(ax, (122.0, 16.2), (124.2, 16.2), color=TEAL, linewidth=1.3)
    box(ax, 124.6, 11.0, 3.0, 10.5, face=TEAL, edge=TEAL, linewidth=1.2, radius=0.7)
    text(ax, 126.1, 16.3, "W", size=10.0, color=WHITE, weight="bold")

    # Attention taps and return path are visually separated from the main stream.
    arrow(
        ax,
        (60.0, 38.6),
        (81.5, 21.9),
        color=TEAL,
        linewidth=1.2,
        linestyle="--",
        connectionstyle="arc3,rad=0.12",
    )
    arrow(
        ax,
        (126.1, 21.9),
        (103.5, 58.0),
        color=TEAL,
        linewidth=1.5,
        connectionstyle="arc3,rad=-0.08",
    )

    operator(ax, 104.5, 58.0, "x", color=ORANGE)
    text(ax, 104.5, 63.0, "Channel reweighting", size=8.2, color=ORANGE, weight="bold")
    arrow(ax, (99.4, 58.0), (102.5, 58.0), color=BLUE, linewidth=1.4)

    box(ax, 109.0, 52.3, 18.0, 11.4, face=ORANGE_FILL, edge=ORANGE, linewidth=1.4, radius=1.0)
    text(ax, 118.0, 60.2, "Attended feature", size=9.5, color=ORANGE, weight="bold")
    text(ax, 118.0, 55.8, "B x 64 x 34 x 49", size=9.2, color=NAVY, weight="bold")
    arrow(ax, (106.5, 58.0), (108.6, 58.0), color=ORANGE, linewidth=1.4)

    panel_label(ax, 132.0, 90.5, "(c)", "Lightweight classification head")
    head_y = 52.3
    head_blocks = [
        (132.0, 16.0, "1 x 8 Conv", "64 -> 98", BLUE_FILL, BLUE),
        (151.0, 16.0, "8 x 1 Conv", "98 -> 98", BLUE_FILL, BLUE),
        (170.0, 11.5, "GAP", "B x 98", PURPLE_FILL, PURPLE),
    ]
    arrow(ax, (127.4, 58.0), (131.6, 58.0), color=LINE, linewidth=1.5)
    for index, (x, width, title, subtitle, face, edge) in enumerate(head_blocks):
        box(ax, x, head_y, width, 11.4, face=face, edge=edge, linewidth=1.4, radius=1.0)
        text(ax, x + width / 2, 60.2, title, size=9.5, color=edge, weight="bold")
        text(ax, x + width / 2, 55.8, subtitle, size=8.8, color=NAVY, weight="bold")
        if index < len(head_blocks) - 1:
            next_x = head_blocks[index + 1][0]
            arrow(ax, (x + width + 0.4, 58.0), (next_x - 0.4, 58.0), color=LINE, linewidth=1.4)

    box(ax, 170.0, 35.5, 11.5, 8.0, face=PURPLE, edge=PURPLE, linewidth=1.4, radius=1.0)
    text(ax, 175.75, 40.6, "Linear", size=9.4, color=WHITE, weight="bold")
    text(ax, 175.75, 37.7, "98 -> 4", size=8.5, color=WHITE)
    arrow(ax, (175.75, 52.0), (175.75, 43.9), color=PURPLE, linewidth=1.4)

    box(ax, 153.0, 19.0, 28.5, 10.0, face=NAVY, edge=NAVY, linewidth=1.4, radius=1.0)
    text(ax, 167.25, 25.9, "Ship class logits", size=10.0, color=WHITE, weight="bold")
    text(ax, 167.25, 21.8, "Cargo | Passenger | Tank | Tug", size=8.0, color=WHITE)
    arrow(ax, (175.75, 35.1), (170.0, 29.4), color=NAVY, linewidth=1.4, connectionstyle="arc3,rad=-0.1")

    text(ax, 2.0, 3.0, "CBA: Conv2D + BatchNorm2D + ReLU", size=8.0, color=GRAY, ha="left")
    text(ax, 182.0, 3.0, "532,166 trainable parameters", size=8.0, color=GRAY, ha="right")

    save_figure(fig, output_dir, "macnna_architecture_main")


def detailed_conv_block(
    ax,
    x: float,
    y: float,
    *,
    layer: str,
    kernel: str,
    channels: str,
    stride: str,
    output: str,
    accent: str,
    face: str,
):
    width = 20.0
    height = 12.0
    box(ax, x, y, width, height, face=face, edge=accent, linewidth=1.25, radius=0.9)
    text(ax, x + 1.6, y + 9.8, layer, size=7.4, color=accent, weight="bold", ha="left")
    text(ax, x + width / 2, y + 7.2, f"Conv {kernel}", size=8.4, color=NAVY, weight="bold")
    text(ax, x + width / 2, y + 4.4, f"C {channels}  |  s {stride}", size=6.5, color=GRAY)
    text(ax, x + width / 2, y + 1.9, output, size=6.9, color=accent, weight="bold")


def draw_detailed_figure(output_dir: Path):
    fig, ax = setup_axes((20.0, 10.0))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)

    box(ax, 3.0, 64.0, 18.0, 22.0, face=PANEL, edge=LIGHT_LINE, linewidth=1.1, radius=1.2)
    text(ax, 12.0, 81.8, "Input", size=9.5, color=NAVY, weight="bold")
    draw_spectrogram_icon(ax, 5.0, 73.0, 14.0, 6.5)
    text(ax, 12.0, 68.0, "log-Mel", size=7.7, color=GRAY)
    text(ax, 12.0, 65.8, "B x 1 x 64 x 94", size=7.5, color=BLUE, weight="bold")

    column_x = [26.0, 49.0, 72.0, 95.0]
    branch_y = [82.0, 66.0, 50.0]
    branch_kernels = [8, 16, 32]
    for row_index, (y, kernel) in enumerate(zip(branch_y, branch_kernels, strict=True)):
        branch_color = [BLUE, TEAL, ORANGE][row_index]
        box(ax, 22.0, y + 2.8, 3.0, 6.4, face=branch_color, edge=branch_color, radius=0.7)
        text(ax, 23.5, y + 6.0, f"k={kernel}", size=6.5, color=WHITE, weight="bold")
        arrow(
            ax,
            (21.4, 75.0),
            (25.6, y + 6.0),
            color=LINE,
            linewidth=1.1,
            connectionstyle="arc3,rad=0.07",
        )

        specs = [
            ("L1", f"1 x {kernel}", "1 -> 32", "(1,2)", "B x 32 x 64 x 48", BLUE, BLUE_FILL),
            ("L2", f"{kernel} x 1", "32 -> 32", "(2,1)", "B x 32 x 33 x 48", TEAL, TEAL_FILL),
            ("L3", f"1 x {kernel}", "32 -> 64", "(1,1)", "B x 64 x 33 x 49", ORANGE, ORANGE_FILL),
            ("L4", f"{kernel} x 1", "64 -> 64", "(1,1)", "B x 64 x 34 x 49", PURPLE, PURPLE_FILL),
        ]
        for index, (layer, conv_kernel, channels, stride, output, accent, face) in enumerate(specs):
            detailed_conv_block(
                ax,
                column_x[index],
                y,
                layer=layer,
                kernel=conv_kernel,
                channels=channels,
                stride=stride,
                output=output,
                accent=accent,
                face=face,
            )
            if index < 3:
                arrow(
                    ax,
                    (column_x[index] + 20.4, y + 6.0),
                    (column_x[index + 1] - 0.4, y + 7.0),
                    color=LINE,
                    linewidth=1.2,
                )

        arrow(
            ax,
            (115.4, y + 6.0),
            (122.2, 72.0),
            color=PURPLE,
            linewidth=1.3,
            connectionstyle="arc3,rad=0.07",
        )

    operator(ax, 125.0, 72.0, "+", color=PURPLE)
    text(ax, 125.0, 78.0, "branch sum", size=7.2, color=PURPLE, weight="bold")
    arrow(ax, (127.0, 72.0), (129.6, 72.0), color=PURPLE, linewidth=1.4)
    box(ax, 130.0, 66.0, 18.0, 12.0, face=PURPLE_FILL, edge=PURPLE, linewidth=1.3, radius=0.9)
    text(ax, 139.0, 73.6, "Fused", size=8.8, color=PURPLE, weight="bold")
    text(ax, 139.0, 69.0, "B x 64 x 34 x 49", size=7.4, color=NAVY, weight="bold")

    box(ax, 3.0, 5.0, 136.0, 35.0, face=PANEL, edge=LIGHT_LINE, linewidth=1.1, radius=1.2)
    text(ax, 6.0, 36.3, "Six-source ECA channel attention", size=9.8, color=TEAL, weight="bold", ha="left")
    text(ax, 6.0, 32.5, "L3 and L4 from each of the three branches", size=7.3, color=GRAY, ha="left")

    attention_blocks = [
        (7.0, 21.0, 20.0, "6 feature maps", "64 channels"),
        (32.0, 21.0, 17.0, "Adaptive GAP", "B x 64 x 1 x 1"),
        (54.0, 21.0, 17.0, "Reshape", "B x 1 x 64"),
        (76.0, 21.0, 20.0, "6 x Conv1D", "k=3, pad=1"),
        (101.0, 21.0, 23.0, "Sigmoid + sum", "B x 64 x 1 x 1"),
    ]
    for index, (x, y, width, title, subtitle) in enumerate(attention_blocks):
        box(ax, x, y - 8.0, width, 11.0, face=TEAL_FILL, edge=TEAL, linewidth=1.2, radius=0.8)
        text(ax, x + width / 2, y, title, size=7.2, color=TEAL, weight="bold")
        text(ax, x + width / 2, y - 3.6, subtitle, size=6.5, color=GRAY)
        if index < len(attention_blocks) - 1:
            arrow(
                ax,
                (x + width + 0.4, y - 2.2),
                (attention_blocks[index + 1][0] - 0.4, y - 2.2),
                color=TEAL,
                linewidth=1.2,
            )

    box(ax, 128.5, 13.0, 6.0, 11.0, face=TEAL, edge=TEAL, linewidth=1.2, radius=0.7)
    text(ax, 131.5, 18.5, "W", size=8.5, color=WHITE, weight="bold")
    arrow(ax, (124.4, 18.8), (128.1, 18.8), color=TEAL, linewidth=1.3)

    operator(ax, 153.0, 72.0, "x", color=ORANGE)
    arrow(ax, (148.4, 72.0), (151.0, 72.0), color=PURPLE, linewidth=1.4)
    arrow(ax, (131.5, 24.4), (152.0, 69.8), color=TEAL, linewidth=1.3, connectionstyle="arc3,rad=0.02")
    box(ax, 157.0, 66.0, 19.0, 12.0, face=ORANGE_FILL, edge=ORANGE, linewidth=1.3, radius=0.9)
    text(ax, 166.5, 73.6, "Attended", size=8.8, color=ORANGE, weight="bold")
    text(ax, 166.5, 69.0, "B x 64 x 34 x 49", size=7.4, color=NAVY, weight="bold")
    arrow(ax, (155.0, 72.0), (156.6, 72.0), color=ORANGE, linewidth=1.4)

    head_blocks = [
        (157.0, 51.0, "Conv 1 x 8", "64 -> 98", BLUE_FILL, BLUE),
        (157.0, 39.0, "Conv 8 x 1", "98 -> 98", BLUE_FILL, BLUE),
        (157.0, 27.0, "Adaptive GAP", "B x 98", PURPLE_FILL, PURPLE),
        (157.0, 15.0, "Linear", "98 -> 4", PURPLE_FILL, PURPLE),
    ]
    arrow(ax, (166.5, 65.6), (166.5, 59.4), color=LINE, linewidth=1.2)
    for index, (x, y, title, subtitle, face, edge) in enumerate(head_blocks):
        box(ax, x, y, 19.0, 8.5, face=face, edge=edge, linewidth=1.15, radius=0.75)
        text(ax, x + 9.5, y + 5.3, title, size=7.0, color=edge, weight="bold")
        text(ax, x + 9.5, y + 2.3, subtitle, size=6.5, color=GRAY)
        if index < len(head_blocks) - 1:
            arrow(ax, (166.5, y - 0.4), (166.5, head_blocks[index + 1][1] + 8.9), color=LINE, linewidth=1.2)

    box(ax, 181.0, 15.0, 16.0, 8.5, face=NAVY, edge=NAVY, linewidth=1.2, radius=0.75)
    text(ax, 189.0, 20.2, "Class logits", size=6.8, color=WHITE, weight="bold")
    text(ax, 189.0, 17.2, "B x 4", size=6.6, color=WHITE)
    arrow(ax, (176.4, 19.2), (180.6, 19.2), color=LINE, linewidth=1.2)

    save_figure(fig, output_dir, "macnna_architecture_detailed")


def draw_overview_figure(output_dir: Path):
    """Draw a presentation-oriented MA-CNN-A overview without tensor dimensions."""
    fig, ax = setup_axes((18.5, 6.4))
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 64)

    box(ax, 4.0, 24.0, 16.0, 16.0, face=PANEL, edge=LIGHT_LINE, linewidth=1.2, radius=1.1)
    draw_spectrogram_icon(ax, 6.5, 29.0, 11.0, 5.8)
    text(ax, 12.0, 26.8, "log-Mel", size=8.7, color=NAVY, weight="bold")
    text(ax, 12.0, 23.7, "spectrogram", size=7.2, color=GRAY)

    box(ax, 28.0, 13.0, 50.0, 39.0, face=PANEL, edge=LIGHT_LINE, linewidth=1.2, radius=1.1)
    text(ax, 53.0, 48.4, "Multi-scale asymmetric backbone", size=9.4, color=NAVY, weight="bold")
    branch_specs = [
        (39.0, BLUE, BLUE_FILL, "k = 8", "short-term features"),
        (29.5, TEAL, TEAL_FILL, "k = 16", "mid-term features"),
        (20.0, ORANGE, ORANGE_FILL, "k = 32", "long-term features"),
    ]
    for y, edge, face, kernel, role in branch_specs:
        box(ax, 31.5, y, 43.0, 7.2, face=face, edge=edge, linewidth=1.1, radius=0.7)
        text(ax, 36.6, y + 4.7, kernel, size=6.9, color=edge, weight="bold")
        text(ax, 57.0, y + 4.7, role, size=6.9, color=NAVY, weight="bold")
        text(ax, 53.0, y + 2.0, "alternating time / frequency convolutions", size=5.8, color=GRAY)

    arrow(ax, (20.4, 32.0), (27.6, 32.0), color=LINE, linewidth=1.4)
    operator(ax, 81.0, 32.0, "+", color=PURPLE)
    text(ax, 81.0, 38.0, "branch sum", size=7.0, color=PURPLE, weight="bold")
    for y in (42.6, 33.1, 23.6):
        arrow(ax, (75.4, y), (79.1, 32.0), color=PURPLE, linewidth=1.2, connectionstyle="arc3,rad=0.04")

    box(ax, 85.0, 25.8, 14.0, 12.4, face=PURPLE_FILL, edge=PURPLE, linewidth=1.2, radius=0.8)
    text(ax, 92.0, 33.8, "Fused", size=8.1, color=PURPLE, weight="bold")
    text(ax, 92.0, 29.8, "feature", size=7.0, color=NAVY)
    arrow(ax, (83.0, 32.0), (84.6, 32.0), color=PURPLE, linewidth=1.4)

    box(ax, 105.0, 20.0, 28.0, 24.0, face=TEAL_FILL, edge=TEAL, linewidth=1.3, radius=1.0)
    text(ax, 119.0, 40.1, "Six-source ECA attention", size=8.3, color=TEAL, weight="bold")
    text(ax, 119.0, 34.6, "L3 and L4 from all branches", size=6.7, color=GRAY)
    text(ax, 119.0, 29.9, "learn channel importance", size=7.0, color=NAVY, weight="bold")
    text(ax, 119.0, 25.5, "reweight fused feature", size=7.0, color=TEAL, weight="bold")
    arrow(ax, (99.4, 32.0), (104.6, 32.0), color=LINE, linewidth=1.4)

    box(ax, 139.0, 24.0, 19.0, 16.0, face=BLUE_FILL, edge=BLUE, linewidth=1.3, radius=0.9)
    text(ax, 148.5, 36.4, "Classification head", size=7.5, color=BLUE, weight="bold")
    text(ax, 148.5, 31.8, "asymmetric Conv", size=6.7, color=NAVY)
    text(ax, 148.5, 28.7, "GAP → Linear", size=6.7, color=NAVY)
    arrow(ax, (133.4, 32.0), (138.6, 32.0), color=LINE, linewidth=1.4)

    box(ax, 164.0, 24.0, 12.0, 16.0, face=NAVY, edge=NAVY, linewidth=1.3, radius=0.9)
    text(ax, 170.0, 35.3, "4 ship", size=7.6, color=WHITE, weight="bold")
    text(ax, 170.0, 31.4, "classes", size=7.6, color=WHITE, weight="bold")
    text(ax, 170.0, 27.8, "logits", size=6.5, color=WHITE)
    arrow(ax, (158.4, 32.0), (163.6, 32.0), color=LINE, linewidth=1.4)

    text(ax, 4.0, 7.3, "All convolution blocks use Conv2D + BatchNorm2D + ReLU.", size=7.2, color=GRAY, ha="left")
    text(ax, 176.0, 7.3, "Cargo · Passenger · Tank · Tug", size=7.2, color=GRAY, ha="right")

    save_figure(fig, output_dir, "macnna_architecture_overview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw publication-style figures for the current MA-CNN-A model.")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    draw_main_figure(args.output_dir)
    draw_detailed_figure(args.output_dir)
    draw_overview_figure(args.output_dir)
