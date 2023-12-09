import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from src.plots.plots import rounded_top_rect, get_font_properties


def plot_overtakes():
    """
        Get all the overtakes since 1999

   """
    df = pd.read_csv('../resources/csv/Overtakes.csv')
    df = df[df['Season'] >= 1999]
    df = df[~df['Race'].str.contains('Sprint')]
    df = df[~df['Race'].str.contains('Season')]

    df = (df.groupby('Season').agg({'Season': 'count', 'Overtakes': 'sum'})
          .rename(columns={'Season': 'Races'}))

    years = df.index.values

    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.xticks(years, font='Fira Sans', fontsize=14, rotation=90)
    bars = ax1.bar(years, df['Overtakes'], color="#AED6F1", edgecolor='white')
    ax1.set_title(f'OVERTAKES IN F1 HISTORY', font='Fira Sans', fontsize=22)
    ax1.set_ylabel('TOTAL OVERTAKES PER SEASON', font='Fira Sans', fontsize=18)
    plt.yticks(font='Fira Sans', fontsize=14, rotation=90)
    ax1.yaxis.grid(False)
    ax1.xaxis.grid(False)

    for bar in bars:
        bar.set_visible(False)

    for bar in bars:
        height = bar.get_height()
        x_value = bar.get_x() + bar.get_width() / 2
        x, y = bar.get_xy()
        width = bar.get_width()
        # Customize color and label based on the x-value
        if x_value < 2010:
            color = '#FFA500'
        elif x_value < 2023:
            color = '#32CD32'
        else:
            color = '#32CD32'
        if x_value == 2005:
            color = '#FF0000'
        if x_value == 2010:
            color = '#FFFF00'

        rounded_box = rounded_top_rect(x, y, width, height, 0.3, color, linewidth=3, y_offset=-7)
        rounded_box.set_facecolor(color)
        ax1.add_patch(rounded_box)
        ax1.text(x_value, height + 10, f'{height}', ha='center', va='bottom', font='Fira Sans', fontsize=16, zorder=3)

    # Create custom legend entries
    legend_entries = [mpatches.Patch(color='orange', label='WITH refueling'),
                      mpatches.Patch(color='red', label='WITH refueling, NO TYRE CHANGES'),
                      mpatches.Patch(color='yellow', label='NO refueling'),
                      mpatches.Patch(color='green', label='NO refueling, WITH DRS')]

    ax2 = ax1.twinx()
    mean_overtakes = round(df['Overtakes'] / df['Races'], 2)
    ax2.plot(years, mean_overtakes, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=8,
             label='Avg overtakes per race in that season', zorder=-3)
    ax2.set_ylabel(f'Avg. overtakes per race (Total Average: {round(mean_overtakes.mean(), 2)})',
                   font='Fira Sans', fontsize=18)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    for label in ax2.get_yticklabels():
        label.set_fontsize(14)  # Change fontsize here
        label.set_fontname('Fira Sans')  # Change font here

    for year, value in mean_overtakes.items():
        text_annotation = ax2.annotate(f'{value}', (year, value), textcoords="offset points", xytext=(0, -20),
                                       ha='center', fontsize=12, color='black')

        # Adding a white edge color to the text
        text_annotation.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])


    # Add a legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2 + legend_entries,
               labels1 + labels2 + [entry.get_label() for entry in legend_entries], loc='upper left',
               prop=get_font_properties('Fira Sans', 'large'))


    plt.tight_layout()
    plt.savefig(f'../PNGs/OVERTAKES IN F1.png', dpi=400)
    plt.show()


def overtakes_by_race():
    df = pd.read_csv('../resources/csv/Overtakes_by_race.csv')
    a = 1