import numpy as np
import six
from matplotlib import pyplot as plt


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=24,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, title="", diff_column=None, col_Widths=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if col_Widths is None:
        col_Widths = [0.15]
        for i in range(len(data.columns) - 1):
            col_Widths.append(0.35)
    mpl_table = ax.table(cellText=data.values, bbox=bbox,
                         colWidths=col_Widths, colLabels=data.columns, **kwargs)

    medal_colors = ['gold', 'silver', '#cd7f32']

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        cell.set_text_props(ha='center')
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            if diff_column is None:
                if k[1] == data.columns.get_loc('Grid change'):
                    value = data.iloc[k[0] - 1, k[1]]
                    if '+' in value:
                        cell.set_facecolor('green')
                    elif value == 'Equal':
                        cell.set_facecolor('gray')
                    else:
                        cell.set_facecolor('red')
                elif k[0] <= 3:
                    cell.set_facecolor(medal_colors[k[0] - 1])
                else:
                    cell.set_facecolor(row_colors[k[0] % len(row_colors)])
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
            if k[1] == len(data.columns) - 1:
                cell.set_text_props(color='w')
                cell.set_edgecolor('black')

    plt.title(title, fontsize=1)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor('black')
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)

    mpl_table.scale(1, 1.25)

    for key, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    return ax

