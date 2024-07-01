import re

import numpy as np
import pandas as pd
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from statsmodels.tsa.arima.model import ARIMA

from src.ergast_api.my_ergast import My_Ergast
from src.utils.utils import append_duplicate_number
from src.variables.team_colors import team_colors_2023, team_colors
import matplotlib.patches as mpatches

from src.variables.variables import point_systems


def race_qualy_avg_metrics(year, session='Q', mode=None):
    """
       Compare points for a team given a threshold

        Parameters:
        year (int): Year to plot
        session (str, optional): Qualy or Race. Default. Q
        predict (bool, optional): Predicts outcome of the season. Default: False
        mode (bool, optional): Total sum or 4 MA. Default: None(4 MA)

   """
    reverse = True
    ergast = My_Ergast()
    data = ergast.get_race_results([year])
    teams = set(data.content[0]['constructorName'])
    team_points = pd.DataFrame(columns=['Team', 'Points', 'Circuit'])
    yticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    if year < 2010:
        yticks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    if session == 'Q':
        data = ergast.get_qualy_results([year])
        yticks = [1, 3, 6, 9, 12, 15, 18, 20]
        title = f'{year} Average qualy position in the last 4 GPs'
        ylabel = 'Qualy position'
        reverse = False
    else:
        ylabel = 'Total points'
        title = f'{year} Average points in the last 4 GPs'

    circuits = []
    for d in data.content:
        circuits.append(d['location'].loc[0])

    circuits = append_duplicate_number(circuits)
    for i in range(len(data.content)):
        for team in teams:
            if session == 'Q':
                points = data.content[i][data.content[i]['constructorName'] == team]['position'].mean()
            else:
                points = data.content[i][data.content[i]['constructorName'] == team]['points'].sum()
            row = [team, points, circuits[i]]
            team_points = team_points._append(pd.Series(row, index=team_points.columns), ignore_index=True)

    team_categories = pd.Categorical(team_points['Team'], categories=team_points['Team'].unique(), ordered=True)
    race_categories = pd.Categorical(team_points['Circuit'], categories=team_points['Circuit'].unique(), ordered=True)
    ct = pd.crosstab(team_categories, race_categories, values=team_points['Points'], aggfunc='sum')
    if mode is not None:
        ct = ct.cumsum(axis=1)
    else:
        ct = ct.rolling(window=4, min_periods=1, axis=1).mean()
    colors_dict = team_colors.get(year, team_colors_2023)
    ordered_colors = [colors_dict[team] for team in ct.index]
    transposed = ct.transpose()


    ax = transposed.plot(figsize=(10, 10), marker='o', color=ordered_colors, markersize=8.5, lw=3)


    font = FontProperties(family='Fira Sans', size=12)
    plt.title(title, font='Fira Sans', fontsize=28)
    plt.xlabel("Races", font='Fira Sans', fontsize=18)
    plt.ylabel(ylabel, font='Fira Sans', fontsize=18)
    last_values = transposed.iloc[-1].values
    handles, labels = ax.get_legend_handles_labels()
    colors = [line.get_color() for line in ax.lines]
    info = list(zip(handles, labels, colors, last_values))
    info.sort(key=lambda item: item[3], reverse=reverse)
    handles, labels, colors, last_values = zip(*info)
    labels = [f"{label} ({last_value:.2f})" for label, last_value in zip(labels, last_values)]

    plt.legend(handles=handles, labels=labels, prop=font, loc="upper left", bbox_to_anchor=(1, 0.6))

    plt.xticks(ticks=range(len(transposed)), labels=transposed.index,
               rotation=90, fontsize=16, fontname='Fira Sans')
    if mode is not None:
        plt.yticks(fontsize=15, fontname='Fira Sans')
    else:
        plt.yticks(yticks, fontsize=15, fontname='Fira Sans')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if session == 'Q':
        ax.invert_yaxis()
    plt.tight_layout()  # Adjusts the plot layout for better visibility
    plt.savefig(f'../PNGs/AVERAGE METRICS {year} - {session}.png', dpi=450)
    plt.show()


def points_per_year(team, mode='team', point_system=2010, start=2014, end=2050, round=None):
    ergast = My_Ergast()
    r = ergast.get_race_results([i for i in range(start, end)])
    points_dict = {year: [] for year in range(start, end)}
    points_positions = point_systems.get(point_system)
    col = 'constructorName' if mode == 'team' else 'fullName'
    for race in r.content:
        race_year = race['year'].loc[0]
        current_round = race['round'].loc[0]
        if round is None or current_round <= round:
            race = race[race[col] == team]
            positions = race['position'].values
            total_points_race = 0
            for p in positions:
                total_points_race += points_positions.get(p, 0)
            points_dict[race_year].append(total_points_race)

    mean_results = {key: np.mean(value) for key, value in points_dict.items()}
    ordered_points = dict(sorted(mean_results.items(), key=lambda item: item[1], reverse=True))
    ordered_years = dict(sorted(mean_results.items(), key=lambda item: item[0], reverse=False))
    rank = 1
    print('--------- MEAN ORDERED BY POINTS ---------')
    for k, v in ordered_points.items():
        print(f'{rank} - {k}: {v:.2f}')
        rank += 1


    print('--------- MEAN ORDERED BY YEARS ---------')
    for k, v in ordered_years.items():
        print(f'{k}: {v:.2f}')


    total_points = {key: np.sum(value) for key, value in points_dict.items()}
    ordered_results = dict(sorted(total_points.items(), key=lambda item: item[1], reverse=True))
    ordered_years = dict(sorted(total_points.items(), key=lambda item: item[0], reverse=False))
    rank = 1
    print('--------- TOTAL ORDERED BY POINTS ---------')
    for k, v in ordered_results.items():
        print(f'{rank} - {k}: {v:.2f}')
        rank += 1
    print('--------- TOTAL ORDERED BY YEAR ---------')
    for k, v in ordered_years.items():
        print(f'{k}: {v:.2f}')
