import fastf1
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from src.plots.plots import round_bars, annotate_bars, get_font_properties
from src.utils.utils import update_name
from src.variables.team_colors import team_colors_2023
from src.plots.plots import lighten_color


def dhl_pitstops(year, groupBy='Driver', round=None, exclude=None, points=False):
    """
        Print pitstops given the dhl data

        Parameters:
        year (int): Year to plot
        groupBy (str): Driver or Team. Default: Driver
        round (int, optional): Plot only a given round. Default: None
        exclude (list, optional): Exclude pit stops. Default: None
        points(bool, optional): Plot DHL system points. Default: False

   """

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    pitstops = pd.read_csv('../resources/csv/Pit_stops.csv', sep='|')
    pitstops = pitstops[pitstops['Year'] == year]
    colors = []
    if round is None:
        if points:
            max_round = pitstops['Race_ID'].max()
            print(
                pitstops[pitstops['Race_ID'] == max_round].groupby('Team')['Points'].sum().sort_values(ascending=False))
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
            colors.append(team_colors_2023[c_data])
        plot_size = (10, 9)
        annotate_fontsize = 16
        y_offset_rounded = -10
        y_offset_annotate = 1
        title = f'{year} DHL PIT STOPS POINTS'
        y_label = 'Points'
        round = 0
        linewidth = 1.75
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
        pitstops = pitstops.sort_values(by='Time', ascending=True)

    if exclude is not None:
        pitstops = pitstops[~pitstops['Driver'].isin(exclude)]

    if points:
        bars = ax1.bar(pitstops['Team'], pitstops['Points'], color=colors,
                       edgecolor='white')
    else:
        bars = ax1.bar(pitstops[groupBy], pitstops['Time'], color=colors,
                       edgecolor='white')

    if groupBy == 'Team':
        colors = [team_colors_2023[i] for i in pitstops['Team'].values]
        annotate_fontsize = 20
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

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
        label.set_fontsize(18)
        label.set_rotation(-25)
        label.set_path_effects([path_effects.withStroke(linewidth=5, foreground=lightened_colors[color_index])])
        color_index += 1

    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim([1.6, ymax])
    plt.tight_layout()
    plt.savefig(f'../PNGs/PIT STOP AVG TIME {year}', dpi=400)
    plt.show()




def fastest_pit_stop_by_team(year):

    pitstops = pd.read_csv('../resources/csv/Pit_stops.csv', sep='|')
    pitstops = pitstops[pitstops['Year'] == year]
    pitstops = pitstops[pitstops['Time'] < 4]
    fastest_pitstops = pitstops.groupby(['Team', 'Race_Name', 'Race_ID'])['Time'].min().reset_index()
    races = pitstops.sort_values(by='Race_ID')['Race_Name'].unique()
    teams = fastest_pitstops['Team'].unique()

    # for team, deltas in mean_delta_by_team.items():
    #     deltas_array = np.array(deltas)
    #     not_nan_indices = ~np.isnan(deltas_array)
    #     plt.plot(np.array(session_names)[not_nan_indices], deltas_array[not_nan_indices],
    #              label=team, marker='o', color=team_colors_2023.get(team), markersize=7, linewidth=3)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    for t in teams:
        team_pit_stops = fastest_pitstops[fastest_pitstops['Team'] == t].sort_values(by='Race_ID')
        team_races = set(team_pit_stops['Race_Name'])
        team_races = [item for item in races if item in team_races]
        plt.plot(team_races, team_pit_stops['Time'],
                     label=t, marker='o', color=team_colors_2023.get(t), markersize=10, linewidth=3)

    plt.title('FASTEST PIT STOP PER TEAM/RACE (ONLY LESS THAN 4s STOPS)',
              font='Fira Sans', fontsize=22)
    plt.ylabel('Time (s)', font='Fira Sans', fontsize=16)
    plt.xticks(rotation=90, font='Fira Sans', fontsize=16)
    plt.yticks(font='Fira Sans', fontsize=16)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(True, linestyle='--', alpha=0.2)
    plt.legend(bbox_to_anchor=(1.0, 0.7), fontsize='large')
    plt.tight_layout()
    plt.savefig('../PNGs/FASTEST PIT STOP TEAM-RACE.png', dpi=450)
    plt.show()