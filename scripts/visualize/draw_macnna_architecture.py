from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT = font_manager.FontProperties(fname=FONT_PATH)
FONT_BOLD = font_manager.FontProperties(fname=FONT_PATH, weight="bold")

COLORS = {
    "ink": "#18324A",
    "muted": "#557086",
    "line": "#8AA2B5",
    "paper": "#F7F4ED",
    "panel": "#FFFFFF",
    "blue": "#2C6EAA",
    "blue_light": "#DCEAF5",
    "cyan": "#2D91A6",
    "cyan_light": "#DDF1F3",
    "gold": "#D99A2B",
    "gold_light": "#F8EBCB",
    "coral": "#D7654D",
    "coral_light": "#F7E2DC",
    "green": "#4A8B72",
    "green_light": "#DFEFE8",
    "purple": "#735D9A",
    "purple_light": "#EAE3F2",
}


def add_text(ax, x, y, text, *, size=8, color=None, ha="center", va="center", bold=False, zorder=6):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        fontproperties=FONT_BOLD if bold else FONT,
        zorder=zorder,
        linespacing=1.25,
    )


def rounded_box(
    ax,
    x,
    y,
    width,
    height,
    *,
    facecolor,
    edgecolor,
    linewidth=1.2,
    radius=1.4,
    linestyle="-",
    zorder=3,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.25,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=None,
    width=1.35,
    style="-|>",
    connectionstyle="arc3",
    linestyle="-",
    zorder=4,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=width,
        color=color or COLORS["line"],
        connectionstyle=connectionstyle,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def operation_circle(ax, x, y, label, *, color):
    circle = Circle((x, y), 2.15, facecolor=color, edgecolor="white", linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    add_text(ax, x, y - 0.1, label, size=12, color="white", bold=True, zorder=7)


def conv_block(ax, x, y, *, title, channels, stride, shape, orientation, facecolor, edgecolor):
    width = 16.2
    height = 13.2
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
        radius=1.3,
    )
    if orientation == "time":
        ax.add_patch(Rectangle((x + 1.4, y + 8.6), 5.5, 1.25, facecolor=edgecolor, edgecolor="none", zorder=5))
        add_text(ax, x + 4.15, y + 11.0, "时间方向", size=5.8, color=edgecolor)
    else:
        ax.add_patch(Rectangle((x + 3.5, y + 7.3), 1.25, 3.8, facecolor=edgecolor, edgecolor="none", zorder=5))
        add_text(ax, x + 4.15, y + 11.7, "频率方向", size=5.8, color=edgecolor)
    add_text(ax, x + 11.3, y + 9.8, title, size=8.5, color=edgecolor, bold=True)
    add_text(ax, x + width / 2, y + 6.2, f"CBA  {channels}", size=7.1, bold=True)
    add_text(ax, x + width / 2, y + 3.9, f"stride={stride}", size=6.2, color=COLORS["muted"])
    add_text(ax, x + width / 2, y + 1.55, shape, size=6.3, color=COLORS["ink"], bold=True)


def draw_input(ax):
    rounded_box(
        ax,
        2.5,
        45.5,
        23.5,
        49.0,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["blue"],
        linewidth=1.5,
        radius=2.0,
    )
    add_text(ax, 14.25, 90.6, "输入与特征", size=10.5, color=COLORS["blue"], bold=True)

    # Compact waveform icon.
    waveform_x = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
    waveform_y = [83.0, 84.5, 81.0, 86.0, 79.8, 84.0, 82.0, 87.0, 78.5, 83.5, 81.2, 85.5, 80.2, 84.3, 82.5, 86.2, 80.8, 83.0]
    ax.plot(waveform_x, waveform_y, color=COLORS["cyan"], linewidth=1.5, zorder=5)
    add_text(ax, 14.25, 77.7, "单通道水声音频  3 s", size=7.5, bold=True)
    add_text(ax, 14.25, 73.9, "重采样至 16 kHz", size=6.6, color=COLORS["muted"])

    arrow(ax, (14.25, 71.4), (14.25, 67.3), color=COLORS["blue"], width=1.2)
    rounded_box(
        ax,
        5.0,
        58.3,
        18.5,
        8.2,
        facecolor=COLORS["blue_light"],
        edgecolor=COLORS["blue"],
        radius=1.0,
    )
    add_text(ax, 14.25, 63.7, "log-Mel 频谱", size=7.7, color=COLORS["blue"], bold=True)
    add_text(ax, 14.25, 60.7, "n_fft=1024 | hop=512 | n_mels=64", size=5.8, color=COLORS["muted"])
    arrow(ax, (14.25, 57.8), (14.25, 53.6), color=COLORS["blue"], width=1.2)
    rounded_box(
        ax,
        5.1,
        47.3,
        18.3,
        5.8,
        facecolor=COLORS["blue"],
        edgecolor=COLORS["blue"],
        radius=0.8,
    )
    add_text(ax, 14.25, 50.2, "B x 1 x 64 x 94", size=8.0, color="white", bold=True)

    # Coordinate directions used throughout the feature maps.
    arrow(ax, (6.0, 42.5), (20.8, 42.5), color=COLORS["coral"], width=1.3)
    add_text(ax, 13.4, 40.0, "时间 W (94 帧)", size=6.2, color=COLORS["coral"])
    arrow(ax, (4.8, 43.5), (4.8, 34.8), color=COLORS["blue"], width=1.3)
    add_text(ax, 2.0, 38.9, "频率 H", size=6.1, color=COLORS["blue"], ha="center", va="center")


def draw_branches(ax):
    branch_rows = [(91.0, 8), (69.5, 16), (48.0, 32)]
    block_x = [33.0, 52.0, 71.0, 90.0]
    block_specs = [
        ("1 x k", "1 -> 32", "(1,2)", "B x 32 x 64 x 48", "time", COLORS["blue_light"], COLORS["blue"]),
        ("k x 1", "32 -> 32", "(2,1)", "B x 32 x 33 x 48", "freq", COLORS["cyan_light"], COLORS["cyan"]),
        ("1 x k", "32 -> 64", "(1,1)", "B x 64 x 33 x 49", "time", COLORS["gold_light"], COLORS["gold"]),
        ("k x 1", "64 -> 64", "(1,1)", "B x 64 x 34 x 49", "freq", COLORS["green_light"], COLORS["green"]),
    ]

    add_text(
        ax,
        29.0,
        109.3,
        "三分支多尺度非对称卷积主干",
        size=11.5,
        color=COLORS["ink"],
        ha="left",
        bold=True,
    )
    add_text(
        ax,
        29.0,
        105.8,
        "每个 CBA = Conv2D + BatchNorm2D + ReLU",
        size=6.8,
        color=COLORS["muted"],
        ha="left",
    )

    for row_y, kernel_size in branch_rows:
        lane_y = row_y - 1.0
        rounded_box(
            ax,
            28.5,
            lane_y - 1.2,
            80.3,
            16.8,
            facecolor="#FFFFFFCC",
            edgecolor="#D5DFE6",
            linewidth=0.9,
            radius=1.8,
        )
        rounded_box(
            ax,
            29.3,
            row_y + 3.2,
            3.0,
            5.0,
            facecolor=COLORS["ink"],
            edgecolor=COLORS["ink"],
            radius=0.7,
            zorder=5,
        )
        add_text(ax, 30.8, row_y + 5.7, f"k\n{kernel_size}", size=6.4, color="white", bold=True, zorder=7)

        # Split the same input tensor into all three branches.
        arrow(
            ax,
            (26.2, 50.2),
            (32.4, row_y + 5.4),
            color=COLORS["line"],
            width=1.05,
            connectionstyle="arc3,rad=0.08",
        )

        for index, (x, spec) in enumerate(zip(block_x, block_specs, strict=True)):
            title, channels, stride, shape, orientation, facecolor, edgecolor = spec
            conv_block(
                ax,
                x,
                row_y,
                title=title.replace("k", str(kernel_size)),
                channels=channels,
                stride=stride,
                shape=shape,
                orientation=orientation,
                facecolor=facecolor,
                edgecolor=edgecolor,
            )
            if index < len(block_x) - 1:
                arrow(ax, (x + 16.4, row_y + 6.6), (block_x[index + 1] - 0.4, row_y + 6.6), width=1.2)

        # The last feature map from each branch participates in feature addition.
        arrow(
            ax,
            (106.5, row_y + 6.6),
            (113.0, 76.1),
            color=COLORS["green"],
            width=1.25,
            connectionstyle="arc3,rad=0.06",
        )


def draw_attention(ax):
    rounded_box(
        ax,
        30.0,
        7.0,
        119.0,
        32.2,
        facecolor="#FFFDFC",
        edgecolor=COLORS["coral"],
        linewidth=1.15,
        radius=2.1,
        linestyle="--",
        zorder=2,
    )
    add_text(ax, 34.0, 35.3, "ECA 风格六路通道注意力", size=10.0, color=COLORS["coral"], ha="left", bold=True)
    add_text(ax, 34.0, 31.7, "3 个分支 x 每分支第 3/4 层特征", size=6.7, color=COLORS["muted"], ha="left")

    # Six feature maps are represented as two grouped source cards per branch.
    source_x = [34.0, 51.0, 68.0]
    branch_colors = [COLORS["blue"], COLORS["cyan"], COLORS["gold"]]
    for x, kernel_size, color in zip(source_x, [8, 16, 32], branch_colors, strict=True):
        rounded_box(
            ax,
            x,
            17.2,
            14.2,
            10.2,
            facecolor="#FFFFFF",
            edgecolor=color,
            linewidth=1.0,
            radius=1.0,
        )
        add_text(ax, x + 7.1, 24.6, f"分支 k={kernel_size}", size=6.7, color=color, bold=True)
        add_text(ax, x + 7.1, 21.7, "L3: 64 x 33 x 49", size=5.7, color=COLORS["ink"])
        add_text(ax, x + 7.1, 19.0, "L4: 64 x 34 x 49", size=5.7, color=COLORS["ink"])

    arrow(ax, (82.8, 22.2), (88.3, 22.2), color=COLORS["coral"], width=1.3)
    rounded_box(
        ax,
        88.7,
        17.0,
        15.5,
        10.5,
        facecolor=COLORS["coral_light"],
        edgecolor=COLORS["coral"],
        radius=1.0,
    )
    add_text(ax, 96.45, 24.4, "各自 GAP", size=7.0, color=COLORS["coral"], bold=True)
    add_text(ax, 96.45, 21.6, "B x 64 x 1 x 1", size=5.8)
    add_text(ax, 96.45, 19.0, "reshape: B x 1 x 64", size=5.4, color=COLORS["muted"])

    arrow(ax, (104.5, 22.2), (109.0, 22.2), color=COLORS["coral"], width=1.3)
    rounded_box(
        ax,
        109.4,
        15.2,
        17.2,
        14.2,
        facecolor=COLORS["purple_light"],
        edgecolor=COLORS["purple"],
        radius=1.1,
    )
    add_text(ax, 118.0, 25.8, "Conv1D", size=7.8, color=COLORS["purple"], bold=True)
    add_text(ax, 118.0, 22.6, "1 -> 1, k=3, p=1", size=5.9)
    add_text(ax, 118.0, 19.8, "Sigmoid", size=6.3, color=COLORS["purple"], bold=True)
    add_text(ax, 118.0, 17.2, "6 组 B x 64 x 1 x 1", size=5.4, color=COLORS["muted"])

    arrow(ax, (126.9, 22.2), (132.0, 22.2), color=COLORS["coral"], width=1.3)
    operation_circle(ax, 134.5, 22.2, "+", color=COLORS["coral"])
    add_text(ax, 134.5, 27.0, "六路权重相加", size=5.8, color=COLORS["coral"], bold=True)
    rounded_box(
        ax,
        138.0,
        17.0,
        8.2,
        10.4,
        facecolor=COLORS["coral"],
        edgecolor=COLORS["coral"],
        radius=1.0,
    )
    add_text(ax, 142.1, 23.7, "W", size=9.0, color="white", bold=True)
    add_text(ax, 142.1, 20.3, "B x 64\nx 1 x 1", size=5.4, color="white", bold=True)

    # Dotted taps show that layers 3 and 4, rather than only the final outputs, feed attention.
    for row_y in [91.0, 69.5, 48.0]:
        arrow(
            ax,
            (79.1, row_y - 0.3),
            (73.0, 39.5),
            color=COLORS["coral"],
            width=0.8,
            style="-",
            connectionstyle="arc3,rad=0.05",
            linestyle=":",
            zorder=1,
        )
        arrow(
            ax,
            (98.1, row_y - 0.3),
            (76.0, 39.5),
            color=COLORS["coral"],
            width=0.8,
            style="-",
            connectionstyle="arc3,rad=-0.05",
            linestyle=":",
            zorder=1,
        )


def draw_fusion_and_head(ax):
    operation_circle(ax, 115.4, 76.1, "+", color=COLORS["green"])
    add_text(ax, 115.4, 82.1, "三分支输出相加", size=6.3, color=COLORS["green"], bold=True)
    arrow(ax, (117.7, 76.1), (120.0, 76.1), color=COLORS["green"], width=1.4)
    rounded_box(
        ax,
        120.4,
        70.7,
        17.0,
        10.8,
        facecolor=COLORS["green_light"],
        edgecolor=COLORS["green"],
        radius=1.1,
    )
    add_text(ax, 128.9, 78.3, "fused", size=7.1, color=COLORS["green"], bold=True)
    add_text(ax, 128.9, 74.3, "B x 64 x 34 x 49", size=6.3, bold=True)

    arrow(ax, (137.8, 76.1), (141.2, 76.1), color=COLORS["green"], width=1.4)
    operation_circle(ax, 143.8, 76.1, "x", color=COLORS["coral"])
    add_text(ax, 143.8, 81.9, "通道重标定", size=6.3, color=COLORS["coral"], bold=True)
    arrow(ax, (142.1, 27.8), (143.8, 73.5), color=COLORS["coral"], width=1.25, connectionstyle="arc3,rad=-0.08")
    arrow(ax, (146.1, 76.1), (148.5, 76.1), color=COLORS["coral"], width=1.4)

    rounded_box(
        ax,
        148.9,
        70.7,
        17.8,
        10.8,
        facecolor=COLORS["coral_light"],
        edgecolor=COLORS["coral"],
        radius=1.1,
    )
    add_text(ax, 157.8, 78.3, "weighted", size=7.1, color=COLORS["coral"], bold=True)
    add_text(ax, 157.8, 74.3, "B x 64 x 34 x 49", size=6.2, bold=True)

    arrow(ax, (167.1, 76.1), (170.3, 76.1), width=1.4)
    conv_block(
        ax,
        170.7,
        69.5,
        title="1 x 8",
        channels="64 -> 98",
        stride="(1,1)",
        shape="B x 98 x 34 x 50",
        orientation="time",
        facecolor=COLORS["blue_light"],
        edgecolor=COLORS["blue"],
    )
    add_text(ax, 178.8, 85.7, "分类头 L1", size=6.3, color=COLORS["blue"], bold=True)

    arrow(ax, (187.2, 76.1), (190.1, 76.1), width=1.4)
    conv_block(
        ax,
        190.5,
        69.5,
        title="8 x 1",
        channels="98 -> 98",
        stride="(1,1)",
        shape="B x 98 x 35 x 50",
        orientation="freq",
        facecolor=COLORS["cyan_light"],
        edgecolor=COLORS["cyan"],
    )
    add_text(ax, 198.6, 85.7, "分类头 L2", size=6.3, color=COLORS["cyan"], bold=True)

    arrow(ax, (207.0, 76.1), (210.3, 76.1), width=1.4)
    rounded_box(
        ax,
        210.7,
        70.7,
        12.2,
        10.8,
        facecolor=COLORS["gold_light"],
        edgecolor=COLORS["gold"],
        radius=1.1,
    )
    add_text(ax, 216.8, 78.5, "GAP", size=8.1, color=COLORS["gold"], bold=True)
    add_text(ax, 216.8, 74.2, "B x 98", size=6.4, bold=True)

    arrow(ax, (223.3, 76.1), (226.2, 76.1), width=1.4)
    rounded_box(
        ax,
        226.6,
        68.3,
        20.8,
        15.6,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=1.4,
    )
    add_text(ax, 237.0, 80.1, "Linear 98 -> 4", size=8.0, color="white", bold=True)
    add_text(ax, 237.0, 76.1, "输出 logits", size=6.2, color="#DCE7EE")
    add_text(ax, 237.0, 71.6, "Cargo | Passenger\nTank | Tug", size=5.9, color="white", bold=True)


def draw_notes(ax):
    rounded_box(
        ax,
        154.0,
        10.0,
        93.0,
        29.2,
        facecolor="#F2F5F7",
        edgecolor="#CBD7DF",
        linewidth=1.0,
        radius=1.8,
        zorder=2,
    )
    add_text(ax, 158.0, 35.0, "读图说明", size=9.5, color=COLORS["ink"], ha="left", bold=True)
    notes = [
        "1. 张量统一写作 B x C x H x W，H 为 Mel 频率，W 为时间帧。",
        "2. 偶数卷积核采用 p=k/2，因此 stride=1 时对应维度会增加 1。",
        "3. ECA 核长由 C=64 自适应得到 k=3；六组 Sigmoid 权重直接相加。",
        "4. 当前 forward 返回 4 维 logits；Softmax 仅在推理解释概率时需要。",
        "5. 当前模型共 532,166 个可训练参数，主干保留 3 个尺度分支。",
    ]
    for index, note in enumerate(notes):
        add_text(ax, 158.0, 30.4 - index * 4.0, note, size=6.2, color=COLORS["muted"], ha="left")


def build_figure(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(25.2, 13.6), dpi=160)
    fig.patch.set_facecolor(COLORS["paper"])
    ax.set_facecolor(COLORS["paper"])
    ax.set_xlim(0, 252)
    ax.set_ylim(0, 124)
    ax.axis("off")

    add_text(
        ax,
        3.0,
        120.0,
        "当前 MA-CNN-A 三分支轻量船舶噪声分类网络",
        size=17.0,
        color=COLORS["ink"],
        ha="left",
        bold=True,
    )
    add_text(
        ax,
        3.0,
        115.2,
        "Actual implementation | multi-scale asymmetric CNN + ECA-style channel attention",
        size=7.7,
        color=COLORS["muted"],
        ha="left",
    )
    ax.plot([3.0, 248.0], [112.4, 112.4], color=COLORS["blue"], linewidth=1.3, zorder=1)

    draw_input(ax)
    draw_branches(ax)
    draw_attention(ax)
    draw_fusion_and_head(ax)
    draw_notes(ax)

    add_text(
        ax,
        248.0,
        4.0,
        "结构依据: src/models/ma_cnn_a.py | 输入依据: src/data/deepship.py",
        size=5.8,
        color=COLORS["muted"],
        ha="right",
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.015)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(
            output_dir / f"macnna_current_architecture.{extension}",
            dpi=200 if extension == "png" else None,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the current MA-CNN-A network architecture.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Directory for SVG, PNG, and PDF outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_figure(args.output_dir)
