import math
import time
from matplotlib import patches
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import os


def lighten_color(hex_color, factor=0.2):

    hex_color = hex_color.strip('#')

    r = int(hex_color[:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)

    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def rounded_top_rect(x, y, width, height, corner_radius, edgecolor, y_offset=0):
    """Create a rectangle path with rounded top."""
    if height >= 0:
        base_y = max(0, y)  # Ensure the starting y value is non-negative
        verts = [
            (x, base_y),  # Bottom left
            (x, max(base_y, base_y + height - corner_radius) + y_offset),  # Start of top-left curve

            # Bezier curves for the top left corner
            (x, max(base_y, base_y + height - corner_radius) + y_offset),
            (x, base_y + height),
            (x + corner_radius, base_y + height),

            # Top straight line
            (x + width - corner_radius, base_y + height),

            # Bezier curves for the top right corner
            (x + width - corner_radius, base_y + height),
            (x + width, base_y + height),
            (x + width, max(base_y, base_y + height - corner_radius) + y_offset),

            # Right straight line and close the polygon
            (x + width, base_y),
            (x, base_y)
        ]
    else:
        y += height
        height = abs(height)
        verts = [
            (x, y + height),  # Top left
            (x, min(0, y + corner_radius)),  # Start of bottom-left curve
            # Bezier curves for the bottom left corner
            (x, min(0, y + corner_radius)),
            (x, y),
            (x + corner_radius, y),
            # Bottom straight line
            (x + width - corner_radius, y),
            # Bezier curves for the bottom right corner
            (x + width - corner_radius, y),
            (x + width, y),
            (x + width, min(0, y + corner_radius)),
            # Right straight line and close the polygon
            (x + width, y + height),
            (x, y + height)
        ]

    codes = [
        patches.Path.MOVETO,
        patches.Path.LINETO,
        patches.Path.CURVE4,
        patches.Path.CURVE4,
        patches.Path.CURVE4,
        patches.Path.LINETO,
        patches.Path.CURVE4,
        patches.Path.CURVE4,
        patches.Path.CURVE4,
        patches.Path.LINETO,
        patches.Path.CLOSEPOLY,
    ]
    path = patches.Path(verts, codes)
    lighter_color = lighten_color(edgecolor, factor=0.3)
    return patches.PathPatch(path, edgecolor=lighter_color)


def round_stacked_bars(x, y, width, height, color):
    corner_radius = min(5 * width, height / 2)

    # Calculate the starting point for the curves based on height
    # Calculate the starting point for the curves based on height
    curve_start_y = y + height * 0.98 - corner_radius
    curve_end_x_left = x + width / 2
    curve_end_x_right = x + width / 2

    # Vertices for the rectangle with rounded top
    verts = [
        (x, y),  # bottom-left
        (x, curve_start_y),  # start of left curve
        (x, y + height),  # Control point for top-left curve
        (curve_end_x_left, y + height),  # end of left curve and start of top-left curve
        (curve_end_x_right, y + height),  # end of top-left curve and start of top-right curve
        (x + width, y + height),  # Control point for top-right curve
        (x + width, curve_start_y),  # end of top-right curve
        (x + width, y),  # bottom-right
        (x, y)  # close polygon
    ]

    codes = [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CLOSEPOLY
    ]

    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=color, edgecolor=color)
    return patch


def stacked_bars(bars, ax, color):
    for bar in bars:
        bar.set_visible(False)

    for bar in bars:
        height = bar.get_height()
        x, y = bar.get_xy()
        width = bar.get_width()
        # Create a fancy bbox with rounded corners and add it to the axes
        rounded_box = round_stacked_bars(x, y, width, height, color)
        rounded_box.set_facecolor(color)
        ax.add_patch(rounded_box)


def round_bars(bars, ax, colors, color_1=None, color_2=None, y_offset_rounded=0, corner_radius=0.1):
    for bar in bars:
        bar.set_visible(False)

    # Overlay rounded rectangle patches on top of the original bars
    i = 0
    for bar in bars:
        height = bar.get_height()
        x, y = bar.get_xy()
        width = bar.get_width()
        if height > 0 and color_1 is not None:
            color = color_1
        elif height < 0 and color_2 is not None:
            color = color_2
        elif type(colors) == str:
            color = colors
        else:
            color = colors[i]

        # Create a fancy bbox with rounded corners and add it to the axes
        rounded_box = rounded_top_rect(x, y, width, height, corner_radius, color, y_offset=y_offset_rounded)
        rounded_box.set_facecolor(color)
        ax.add_patch(rounded_box)
        i += 1


def annotate_bars(bars, ax, y_offset_annotate, annotate_fontsize, text_annotate='default', ceil_values=False):
    for bar in bars:
        height = bar.get_height()
        if ceil_values:
            height = math.ceil(height)
        if height < 0:
            y_offset = -y_offset_annotate - 0.01
        else:
            y_offset = y_offset_annotate
        if height != 0:
            plot_text = text_annotate_bars(height, text_annotate)
            ax.text(bar.get_x() + bar.get_width() / 2, height + y_offset, plot_text, ha='center', va='bottom',
                    font='Fira Sans', fontsize=annotate_fontsize)
        else:
            bar.set_visible(False)


def text_annotate_bars(height, original_text):
    if original_text == 'default':
        return f'{height}'
    else:
        if height < 0:
            original_text = original_text.replace('+', '')
        return original_text.replace('{height}', str(height))


def get_handels_labels(ax):
    return ax.get_legend_handles_labels()


def get_font_properties(family, size):
    return FontProperties(family=family, size=size)


def title_and_labels(plt, title, title_fontsize, x_label, x_fontsize, y_label, y_fontsize, title_offset=0):
    plt.title(title, font='Fira Sans', fontsize=title_fontsize, x=title_offset)
    plt.xlabel(x_label, font='Fira Sans', fontsize=x_fontsize)
    plt.ylabel(y_label, font='Fira Sans', fontsize=y_fontsize)
