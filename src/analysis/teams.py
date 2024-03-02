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


def race_qualy_avg_metrics(year, session='Q', predict=False, mode=None):
    """
       Compare points for a team given a threshold

        Parameters:
        year (int): Year to plot
        session (str, optional): Qualy or Race. Default. Q
        predict (bool, optional): Predicts outcome of the season. Default: False
        mode (bool, optional): Total sum or 4 MA. Default: None(4 MA)

   """
    reverse = True
    ergast = Ergast()
    data = ergast.get_race_results(season=year, limit=1000)
    teams = set(data.content[0]['constructorName'])
    team_points = pd.DataFrame(columns=['Team', 'Points', 'Circuit'])
    yticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    if year < 2010:
        yticks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    if session == 'Q':
        data = ergast.get_qualifying_results(season=year, limit=1000)
        yticks = [1, 3, 6, 9, 12, 15, 18, 20]
        title = f'{year} Average qualy position in the last 4 GPs'
        ylabel = 'Qualy position'
        reverse = False
    else:
        ylabel = 'Total points'
        if predict:
            title = f'{year} Average points prediction in the last 4 GPs'
        else:
            title = f'{year} Average points in the last 4 GPs'

    circuits = np.array(data.description['circuitId'])
    circuits = [i.replace('_', ' ').title() for i in circuits]
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
    if predict:
        forecasted_data = []
        for team in transposed.columns:
            model = ARIMA(transposed[team], order=(5, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=6)
            forecasted_data.append(forecast)

        forecasted_df = pd.DataFrame(forecasted_data).transpose()
        forecasted_df.columns = transposed.columns

        circuits = ergast.get_race_schedule(season=year, limit=1000)['circuitId'].values
        start_index = len(transposed)
        end_index = start_index + len(forecasted_df)
        new_indices = [circuits[i].title().replace('_', ' ') for i in range(start_index, end_index)]
        forecasted_df.index = new_indices

        transposed = pd.concat([transposed, forecasted_df])
        transposed = transposed.where(transposed >= 0, 0)

    ax = transposed.plot(figsize=(12, 12), marker='o', color=ordered_colors, markersize=7, lw=3)

    if predict:
        start_x = len(transposed) - len(forecasted_df)  # Start of forecast
        end_x = len(transposed) - 1  # End of forecast (which is essentially the length of your combined data)
        ax.axvspan(start_x, end_x, facecolor='green', alpha=0.2)
        ax.annotate('Predictions', xy=((start_x + end_x) / 2, ax.get_ylim()[1] - 1),
                    xycoords='data', ha='center', fontsize=16, color='black', alpha=0.7,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))

    font = FontProperties(family='Fira Sans', size=12)
    plt.title(title, font='Fira Sans', fontsize=28)
    plt.xlabel("Races", font='Fira Sans', fontsize=18)
    plt.ylabel(ylabel, font='Fira Sans', fontsize=18)
    predict_patch = mpatches.Patch(color='green', alpha=0.5, label='Predictions')
    last_values = transposed.iloc[-1].values
    handles, labels = ax.get_legend_handles_labels()
    colors = [line.get_color() for line in ax.lines]
    info = list(zip(handles, labels, colors, last_values))
    info.sort(key=lambda item: item[3], reverse=reverse)
    handles, labels, colors, last_values = zip(*info)
    labels = [f"{label} ({last_value:.2f})" for label, last_value in zip(labels, last_values)]

    if predict:
        handles = list(handles)
        handles.append(predict_patch)
        labels.append("Predictions")

    plt.legend(handles=handles, labels=labels, prop=font, loc="upper left", bbox_to_anchor=(1, 0.6))

    plt.xticks(ticks=range(len(transposed)), labels=transposed.index,
               rotation=90, fontsize=12, fontname='Fira Sans')
    if mode is not None:
        plt.yticks(fontsize=12, fontname='Fira Sans')
    else:
        plt.yticks(yticks, fontsize=12, fontname='Fira Sans')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if session == 'Q':
        ax.invert_yaxis()
    plt.tight_layout()  # Adjusts the plot layout for better visibility
    plt.savefig(f'../PNGs/AVERAGE POINTS {year}.png', dpi=450)
    plt.show()


def points_per_year(team, mode='team', point_system=2010, start=2014, end=2024):
    ergast = My_Ergast()
    r = ergast.get_race_results([i for i in range(start, end)])
    points_dict = {year: [] for year in range(start, end)}
    points_positions = point_systems.get(point_system)
    col = 'constructorName' if mode == 'team' else 'fullName'
    for race in r.content:
        race_year = race['year'].loc[0]
        race = race[race[col] == team]
        positions = race['position'].values
        total_points_race = 0
        for p in positions:
            total_points_race += points_positions.get(p, 0)
        points_dict[race_year].append(total_points_race)

    mean_results = {key: np.mean(value) for key, value in points_dict.items()}
    ordered_results = dict(sorted(mean_results.items(), key=lambda item: item[1], reverse=True))
    rank = 1
    for k, v in ordered_results.items():
        print(f'{rank} - {k}: {v:.2f}')
        rank += 1
