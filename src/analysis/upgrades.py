import pandas as pd
from matplotlib import ticker, pyplot as plt

from src.plots.plots import get_handels_labels, get_font_properties, title_and_labels
from src.variables.team_colors import team_colors_2023, team_colors


def plot_upgrades(year, scope=None):
    """
       Plot the upgrades for a season

        Parameters:
        scope (str, optional): (Performance|Circuit Specific)

   """

    upgrades = pd.read_csv('../resources/csv/Upgrades.csv', sep='|')
    upgrades = upgrades[upgrades['Year'] == year]
    all_races_ordered = upgrades[upgrades['Round'] != 1].sort_values('Round')['Race'].unique()
    if scope is not None:
        upgrades = upgrades[upgrades['Reason'] == scope]
    upgrades = upgrades[upgrades['Race'] != 'Bahrain']  # Example: Bahrain is not present

    # Make 'Team' and 'Race' categorical with the full set of unique values
    upgrades['Team'] = pd.Categorical(upgrades['Team'], categories=upgrades['Team'].unique(), ordered=True)
    upgrades['Race'] = pd.Categorical(upgrades['Race'], categories=all_races_ordered, ordered=True)

    # Create the crosstab and calculate the cumulative sum
    ct = pd.crosstab(upgrades['Team'], upgrades['Race'], dropna=False)

    # Reindex the crosstab to include all rounds, filling with previous valid value or 0 if none
    cumulative_sum = ct.reindex(columns=all_races_ordered, fill_value=0).fillna(method='ffill', axis=1)
    cumulative_sum = cumulative_sum.cumsum(axis=1)
    ordered_colors = [team_colors[year][team] for team in cumulative_sum.index]
    transposed = cumulative_sum.transpose()
    last_values = transposed.iloc[-1].values
    font = get_font_properties('Fira Sans', 12)

    if scope is None:
        scope = ''
    else:
        scope += ' '

    ax = transposed.plot(figsize=(10, 10), marker='o', color=ordered_colors, markersize=7.25, lw=4)

    title_and_labels(plt, f'Cumulative {scope}Upgrades for Each Team', 28,
                     'Races', 18, 'Number of Upgrades', 18, 0.5)

    handles, labels = get_handels_labels(ax)
    colors = [line.get_color() for line in ax.lines]
    info = list(zip(handles, labels, colors, last_values))
    info.sort(key=lambda item: item[3], reverse=True)
    handles, labels, colors, last_values = zip(*info)
    labels = [f"{label} ({last_value:.0f})" for label, last_value in zip(labels, last_values)]

    plt.legend(handles=handles, labels=labels, prop=font, loc="upper left", fontsize='x-large')
    plt.xticks(ticks=range(len(transposed)), labels=transposed.index,
               rotation=90, fontsize=12, fontname='Fira Sans')
    plt.yticks(fontsize=12, fontname='Fira Sans')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f'../PNGs/{scope} UPGRADES.png', dpi=400)
    plt.show()
