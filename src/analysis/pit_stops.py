import statistics

import fastf1
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects

from src.ergast_api.my_ergast import My_Ergast
from src.plots.plots import round_bars, annotate_bars, get_font_properties
from src.utils.utils import update_name, name_count, restart_name_count
from src.variables.team_colors import team_colors_2023, team_colors
from src.plots.plots import lighten_color


def dhl_pitstops(year, groupBy='Driver', round=None, points=False):
    """
        Print pitstops given the dhl data

        Parameters:
        year (int): Year to plot
        groupBy (str): Driver or Team. Default: Driver
        round (int, optional): Plot only a given round. Default: None
        exclude (list, optional): Exclude pit stops. Default: None
        points(bool, optional): Plot DHL system points. Default: False

   """
    pitstops = pd.read_csv('../resources/csv/Pit_stops.csv')
    pitstops = pitstops[pitstops['Year'] == year]
    colors = []
    if round is None:
        if points:
            max_round = pitstops['Race_ID'].max()
            pitstops = pitstops.groupby('Team')['Points'].sum()
        else:
            pitstops = pitstops.groupby(groupBy)['Time'].median()
    else:
        pitstops = pitstops[pitstops['Race_ID'] == round]
    pitstops = pitstops.reset_index()
    if points:
        pitstops = pitstops.sort_values(by='Points', ascending=False)
        color_data = [i for i in pitstops['Team']]
        for c_data in color_data:
            colors.append(team_colors[year][c_data])
        plot_size = (10, 9)
        annotate_fontsize = 26
        y_offset_rounded = -10
        y_offset_annotate = 1
        title = f'{year} DHL PIT STOPS POINTS'
        y_label = 'Points'
        round = 0
        linewidth = 3.75
    else:
        pitstops = pitstops.sort_values(by='Time', ascending=True)
        pitstops['Time'] = pitstops['Time'].round(2)
        drivers = [i for i in pitstops[groupBy]]
        if groupBy == 'Driver':
            plot_size = (17, 10)
        else:
            plot_size = (12, 10)
        annotate_fontsize = 12
        y_offset_rounded = 0.06
        y_offset_annotate = 0.05
        title = f'{year} MEDIAN PIT STOP TIMES'
        if round is not None:
            title = 'PIT STOPS TIME'
        y_label = 'Time (s)'
        round = 2
        linewidth = 4

        for driver in drivers:
            for key, value in fastf1.plotting.DRIVER_COLORS.items():
                parts = key.split(" ", 1)
                new_key = parts[1] if len(parts) > 1 else key
                if (new_key == driver.lower()) or (new_key == 'guanyu' and driver == 'Zhou'):
                    colors.append(value)
                    break

    fig, ax1 = plt.subplots(figsize=plot_size)

    if round is not None and groupBy == 'Driver':
        pitstops = pitstops.sort_values(by='Lap', ascending=True)
        pitstops['Driver'] = pitstops['Driver'].apply(update_name)
        restart_name_count()
        pitstops = pitstops.sort_values(by='Time', ascending=True)

    if groupBy == 'Team':
        pitstops = pitstops.groupby(groupBy)['Time'].median().reset_index()
        colors = [team_colors.get(year)[i] for i in pitstops['Team'].values]
        annotate_fontsize = 20
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    if points:
        bars = ax1.bar(pitstops['Team'], pitstops['Points'], color=colors,
                       edgecolor='white')
    else:
        bars = ax1.bar(pitstops[groupBy], pitstops['Time'], color=colors,
                       edgecolor='white')

    round_bars(bars, ax1, colors, y_offset_rounded=y_offset_rounded, linewidth=linewidth)
    annotate_bars(bars, ax1, y_offset_annotate, annotate_fontsize, round=round)

    ax1.set_title(title, font='Fira Sans', fontsize=28)
    ax1.set_ylabel(y_label, font='Fira Sans', fontweight='bold', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=18)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(False)
    font_properties = get_font_properties('Fira Sans', 14)

    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_properties)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_properties)

    color_index = 0
    lightened_colors = [lighten_color(i) for i in colors]
    for label in ax1.get_xticklabels():
        label.set_color('black')
        label.set_fontsize(16)
        label.set_rotation(-25)
        label.set_path_effects([path_effects.withStroke(linewidth=5, foreground=lightened_colors[color_index])])
        color_index += 1

    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim([1.6, ymax])
    plt.tight_layout()
    plt.savefig(f'../PNGs/PIT STOP AVG TIME {year}', dpi=400)
    plt.show()




def fastest_pit_stop_by_team(year):
    pitstops = pd.read_csv('../resources/csv/Pit_stops.csv', sep=',')
    pitstops = pitstops[pitstops['Year'] == year]
    pitstops = pitstops[pitstops['Time'] <= 4]
    fastest_pitstops = pitstops.groupby(['Team', 'Race_Name', 'Race_ID'])['Time'].min().reset_index()
    races_df = pitstops[['Race_ID', 'Race_Name']].drop_duplicates().sort_values(by='Race_ID')

    fig, ax1 = plt.subplots(figsize=(10, 8))
    for t in fastest_pitstops['Team'].unique():
        team_data = fastest_pitstops[fastest_pitstops['Team'] == t]
        team_data = races_df.merge(team_data, on='Race_Name', how='left')
        x = team_data['Race_Name']
        y = team_data['Time']
        plt.plot(x, y, label=t, marker='o', linestyle='-', color=team_colors[year].get(t, '#000000'), markersize=7.5,
                 linewidth=4.25)
        plt.plot(x, y, marker='o', linestyle='', color=team_colors[year].get(t, '#000000'),
                 markersize=7.5)


    plt.title(f'FASTEST PIT STOP PER TEAM/RACE (ONLY LESS THAN 4s STOPS) IN {year}',
              font='Fira Sans', fontsize=22)
    plt.ylabel('Time (s)', font='Fira Sans', fontsize=16)
    plt.xticks(rotation=90, font='Fira Sans', fontsize=16)
    plt.yticks(font='Fira Sans', fontsize=16)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(True, linestyle='--', alpha=0.2)
    plt.legend(bbox_to_anchor=(1.0, 0.7), fontsize='large')
    plt.tight_layout()
    plt.savefig(f'../PNGs/FASTEST PIT STOP TEAM-RACE {year}.png', dpi=450)
    plt.show()



def pitstops_per_year(year):

    pitstops = pd.read_csv('../resources/csv/Pit_stops.csv')
    pitstops = pitstops[pitstops['Year'] == year].sort_values(by='Race_ID')
    pitstops_year = pitstops.groupby(['Race_Name', 'Race_ID']).size().reset_index().sort_values(by='Race_ID')
    pitstops_year.columns = ['Race', 'Id', 'Count']
    fig, ax = plt.subplots(figsize=(8,8))
    bars = plt.bar(pitstops_year['Race'], pitstops_year['Count'])
    round_bars(bars, ax, '#fc6600', color_1=None, color_2=None, y_offset_rounded=0, corner_radius=0.1, linewidth=2.5)
    annotate_bars(bars, ax, 0.25, 14, text_annotate='default', ceil_values=False, round=0,
                  y_negative_offset=0.04, annotate_zero=False, negative_offset=0)

    plt.title('Number of pit stops per race', font='Fira Sans', fontsize=24)
    plt.xlabel('Race', font='Fira Sans', fontsize=18)
    plt.ylabel('Number of pit stops', font='Fira Sans', fontsize=18)
    plt.xticks(rotation=90, font='Fira Sans', fontsize=15)
    plt.yticks(font='Fira Sans', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'../PNGs/PIT STOPS IN {year}.png', dpi=450)
    plt.show()



def pitstops_pirelli_era(end=2025):

    ergast = My_Ergast()
    pitstops_ergast = ergast.get_pit_stops([i for i in range(2011, 2023)]).content
    races = ergast.get_race_results([i for i in range(2011, 2023)]).content
    pit_stop_dict = {}

    for p, r in zip(pitstops_ergast, races):
        year = p['year'].loc[0]
        r = r[~r['status'].isin(['Withdrew', 'Did not start'])]
        mean_pit_stops_race = len(p) / len(r)
        if year not in pit_stop_dict:
            pit_stop_dict[year] = [mean_pit_stops_race]
        else:
            pit_stop_dict[year].append(mean_pit_stops_race)

    mean_pit_stops = []
    years = [i for i in range(2011, end)]
    for y, p in pit_stop_dict.items():
        mean_pit_stops.append(statistics.mean(p))

    pitstops_dhl = pd.read_csv('../resources/csv/Pit_stops.csv', sep=',')
    for i in range(2023, end):
        pitstops_year= pitstops_dhl[pitstops_dhl['Year'] == i].sort_values(by='Race_ID')
        mean_pit_stops.append((pitstops_year.groupby('Race_ID').size()/20).mean())

    fig, ax = plt.subplots(figsize=(9, 8))
    bars = plt.bar(years, mean_pit_stops)
    round_bars(bars, ax, '#00BFFF', color_1=None, color_2=None, y_offset_rounded=0.05, corner_radius=0.1, linewidth=4)
    annotate_bars(bars, ax, 0.01, 18, text_annotate='default', ceil_values=False, round=2,
                  y_negative_offset=0.04, annotate_zero=False, negative_offset=0)

    plt.title('Average pit stops per driver/season in the Pirelli era', font='Fira Sans', fontsize=24)
    plt.xlabel('Year', font='Fira Sans', fontsize=20)
    plt.ylabel('Average pit stops per driver', font='Fira Sans', fontsize=20)
    plt.xticks(years, rotation=35, font='Fira Sans', fontsize=18)
    plt.yticks(font='Fira Sans', fontsize=16)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'../PNGs/PIT STOPS IN PIRELLI ERA.png', dpi=450)
    plt.show()