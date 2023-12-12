import statistics

import fastf1
import numpy as np
import pandas as pd
from fastf1.core import Laps
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt
from fastf1 import utils, plotting
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from timple.timedelta import strftimedelta
import matplotlib.patches as mpatches
from src.ergast_api.my_ergast import My_Ergast
from src.exceptions import qualy_by_year
from src.exceptions.custom_exceptions import QualyException
from src.plots.plots import rounded_top_rect, round_bars, annotate_bars
from src.utils.utils import append_duplicate_number
from src.variables.team_colors import team_colors_2023


def team_performance_vs_qualy_last_year(team, delete_circuits=[], year=2023):
    """
       Plot the performance of a team against last year qualy sessions

       Parameters:
       team (str): Team to analyze
       delete_circuits (array, optional): Circuits to exclude from the analysis. Default = []
       year (int, optional): Year of the analysis. Default = 2023

   """

    ergast = Ergast()
    prev_year = ergast.get_qualifying_results(season=year - 1, limit=1000)
    current_year = ergast.get_qualifying_results(season=year, limit=1000)

    prev_cir = prev_year.description['raceName'].values
    current_cir = current_year.description['raceName'].values

    def intersect_lists_ordered(list1, list2):
        return [item for item in list1 if item in list2]

    result = intersect_lists_ordered(current_cir, prev_cir)
    race_index_prev = []

    for item in result:
        if item in prev_year.description['raceName'].values:
            race_index_prev.append(prev_year.description['raceName'].to_list().index(item) + 1)

    race_index_current = []
    for i, item in current_year.description['raceName'].items():
        if item in result:
            race_index_current.append(i + 1)

    delta = []
    color = []
    result = [track.replace('_', ' ').title() for track in result]
    result = [track.split(' ')[0] for track in result]
    for i in range(len(race_index_current)):
        prev_year = fastf1.get_session(year - 1, race_index_prev[i], 'Q')
        prev_year.load()
        current_year = fastf1.get_session(year, race_index_current[i], 'Q')
        current_year.load()

        fast_prev_year = prev_year.laps.pick_team(team).pick_fastest()['LapTime']
        fast_current_year = current_year.laps.pick_team(team).pick_fastest()['LapTime']
        if result[i] in delete_circuits:
            delta.append(np.nan)
            color.append('#FF0000')
        else:
            delta_time = round(fast_current_year.total_seconds() - fast_prev_year.total_seconds(), 3)
            delta.append(delta_time)
            if delta_time > 0:
                color.append('#FF0000')
            else:
                color.append('#008000')

    fig, ax = plt.subplots(figsize=(24, 10))
    bars = ax.bar(result, delta, color=color)

    for bar in bars:
        bar.set_visible(False)
    i = 0
    for bar in bars:
        height = bar.get_height()
        x, y = bar.get_xy()
        width = bar.get_width()
        # Create a fancy bbox with rounded corners and add it to the axes
        rounded_box = rounded_top_rect(x, y, width, height, 0.1, color[i])
        rounded_box.set_facecolor(color[i])
        ax.add_patch(rounded_box)
        i += 1

    for i in range(len(delta)):
        if delta[i] > 0:  # If the bar is above y=0
            plt.text(result[i], delta[i] + 0.4, '+' + str(delta[i]) + 's',
                     ha='center', va='top', font='Fira Sans', fontsize=12)
        else:  # If the bar is below y=0
            plt.text(result[i], delta[i] - 0.4, str(delta[i]) + 's',
                     ha='center', va='bottom', font='Fira Sans', fontsize=12)

    plt.axhline(0, color='white', linewidth=0.8)
    differences_series = pd.Series(delta)
    # Calculate the rolling mean
    mean_y = differences_series.rolling(window=4, min_periods=1).mean().tolist()
    plt.plot(result, mean_y, color='red',
             marker='o', markersize=5.5, linewidth=3.5, label='Moving Average (4 last races)')

    # Add legend patches for the bar colors
    red_patch = mpatches.Patch(color='red', label=f'{year - 1} Faster')
    green_patch = mpatches.Patch(color='green', label=f'{year} Faster')
    plt.legend(handles=[green_patch, red_patch,
                        plt.Line2D([], [], color='orange', marker='o', markersize=5.5, linestyle='',
                                   label='Moving Average (4 last circuits)')], fontsize='x-large', loc='upper left')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')
    plt.title(f'{team.upper()} QUALY COMPARISON: 2022 vs. 2023', font='Fira Sans', fontsize=24)
    plt.xlabel('Circuit', font='Fira Sans', fontsize=16)
    plt.ylabel('Time diff (seconds)', font='Fira Sans', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{team} QUALY COMPARATION {year - 1} vs {year}.png', dpi=450)
    plt.show()


def qualy_results(session):
    """
       Plot the results of a qualy with fastF1 API

       Parameters:
       session (Session): Session of the lap

    """

    drivers = pd.unique(session.laps['Driver'])
    list_fastest_laps = list()
    for drv in drivers:
        drvs_fastest_lap = session.laps.pick_driver(drv).pick_fastest()
        list_fastest_laps.append(drvs_fastest_lap)
    fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

    pole_lap = fastest_laps.pick_fastest()
    fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']
    fastest_laps.dropna(how='all', inplace=True)

    team_colors = list()
    for index, lap in fastest_laps.iterlaps():
        if lap['Team'] == 'Sauber':
            color = '#FD8484'
        else:
            color = plotting.team_color(lap['Team'])
        team_colors.append(color)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'],
            color=team_colors, edgecolor='grey')

    ax.set_yticks(fastest_laps.index)
    ax.set_yticklabels(fastest_laps['Driver'])
    ax.invert_yaxis()

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)
    lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')

    plt.title(f"{session.event['EventName']} {session.event.year} Qualifying\n"
                 f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})", font='Fira Sans', fontsize=20)

    plt.xlabel("Diff in seconds (s)", font='Fira Sans', fontsize=17)
    plt.ylabel("Driver", font='Fira Sans', fontsize=17)

    def custom_formatter(x, pos):
        return round(x * 100000, 1)

    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax.xaxis.grid(True, color='white', linestyle='--')
    plt.xticks(font='Fira Sans', fontsize=15)
    plt.yticks(font='Fira Sans', fontsize=15)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.savefig(f"../PNGs/QUALY OVERVIEW {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def qualy_diff(year):
    """
       Plot the qualy time diff between 2 teams

       Parameters:
       team_1 (str): Team 1
       team_2 (str): Team 2
       rounds(int): Number of rounds to be analyzed

    """

    session_names = []
    n_qualys = Ergast().get_qualifying_results(season=year, limit=1000).content
    delta_diff = {}
    for i in range(len(n_qualys)):
        qualy_delta_diffs = {}
        session = fastf1.get_session(year, i + 1, 'Q')
        session.load(telemetry=True)
        session_names.append(session.event['Location'].split('-')[0])
        from src.utils.utils import call_function_from_module
        try:
            call_function_from_module(qualy_by_year, f"year_{year}", i + 1)
            teams_session_dict = {}
            teams_session = []
            for i in range(2, -1, -1):
                q_session = session.laps.split_qualifying_sessions()[i]
                teams_session = q_session['Team'].unique()
                for t in teams_session:
                    if t not in teams_session_dict:
                        teams_session_dict[t] = i

            q_session = session.laps.pick_teams(teams_session)
            fastest_laps = pd.DataFrame(q_session.loc[q_session.groupby('Team')['LapTime'].idxmin()]
                                        [['Team', 'LapTime']]).sort_values(by='LapTime')
            fastest_lap_in_session = fastest_laps['LapTime'].iloc[0]
            fastest_laps['DeltaPercent'] = ((fastest_laps['LapTime'] - fastest_lap_in_session)
                                            / fastest_lap_in_session) * 100

            for t in teams_session:
                if t not in qualy_delta_diffs:
                    qualy_delta_diffs[t] = fastest_laps[fastest_laps['Team'] == t]['DeltaPercent'].loc[0]

            for t, v in qualy_delta_diffs.items():
                if t not in delta_diff:
                    delta_diff[t] = [v]
                else:
                    delta_diff[t].append(v)
        except QualyException:
            teams = session.laps['Team'].unique()
            for t in teams:
                if t not in delta_diff:
                    delta_diff[t] = [np.NaN]
                else:
                    delta_diff[t].append(np.NaN)

    session_names = append_duplicate_number(session_names)

    fig, ax1 = plt.subplots(figsize=(12, 10))
    plt.rcParams["font.family"] = "Fira Sans"
    for team, deltas in delta_diff.items():
        plt.plot(session_names, deltas, label=team, marker='o',
                 color=team_colors_2023.get(team), markersize=7, linewidth=3)

        # Find the indices right before and after the NaNs to plot dashed lines
        for i, delta in enumerate(deltas):
            if np.isnan(delta):
                # Find the previous non-NaN index
                prev_index = max([j for j in range(i - 1, -1, -1) if not np.isnan(deltas[j])], default=None)
                # Find the next non-NaN index
                next_index = min([j for j in range(i + 1, len(deltas)) if not np.isnan(deltas[j])], default=None)

                # If both indices are found, plot a dashed line between them
                if prev_index is not None and next_index is not None:
                    plt.plot([session_names[prev_index], session_names[next_index]],
                             [deltas[prev_index], deltas[next_index]],
                             color=team_colors_2023.get(team), linewidth=1.25, linestyle='--')

    plt.gca().invert_yaxis()
    plt.legend(loc='lower right', fontsize='large')
    plt.title(f'{year} AVERAGE QUALY DIFF PER CIRCUIT', font='Fira Sans', fontsize=24)
    plt.ylabel('Percentage time difference (%)', font='Fira Sans', fontsize=20)
    plt.xlabel('Circuit', font='Fira Sans', fontsize=20)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(True, linestyle='--', alpha=0.2)
    plt.xticks(rotation=90, fontsize=15, fontname='Fira Sans')
    plt.yticks(fontsize=15, fontname='Fira Sans')
    plt.tight_layout()
    plt.savefig(f"../PNGs/{year} ONE LAP PACE DIFFERENCE.png", dpi=400)
    plt.show()


def qualy_margin(circuit, start=None, end=None):
    """
       Prints the qualy margins in a given circuits

       Parameters:
       circuit(str): Circuit to analyze
       start (int, optional): Year of start. Default: 1950
       start (int, optional): Year of end. Default: 2024

    """

    ergast = My_Ergast()
    dict_years = {}
    dict_drivers = {}
    if start is None:
        start = 1950
    if end is None:
        end = 2024
    qualy = ergast.get_qualy_results([i for i in range(start, end)])
    for q in qualy.content:
        q_cir = q['circuitRef'].min()
        if q_cir == circuit:
            year = q['year'].min()
            try:
                pole_time = q['q3'].values[0]
                second_time = q['q3'].values[1]
                if pd.isna(pole_time) or pd.isna(pole_time):
                    raise Exception
            except:
                try:
                    pole_time = q['q2'].values[0]
                    second_time = q['q2'].values[1]
                    if pd.isna(pole_time) or pd.isna(pole_time):
                        raise Exception
                except:
                    pole_time = q['q1'].values[0]
                    second_time = q['q1'].values[1]
            diff = second_time - pole_time
            dict_years[year] = diff / np.timedelta64(1, 's')
            dict_drivers[year] = f'from {q["familyName"].values[0]} to {q["familyName"].values[1]}'

    dict_years = {k: v for k, v in sorted(dict_years.items(), key=lambda item: item[1], reverse=False)}

    for key, value in dict_years.items():
        drivers = dict_drivers[key]
        print(f'{key}: {value}s {drivers}')


def qualy_diff_teammates(team, rounds):
    """
         Prints the qualy diff between teammates

         Parameters:
         team (str): Team
         rounds (int): Rounds to be analyzed

    """
    from src.utils.utils import call_function_from_module
    circuits = []
    legend = []
    color = []
    differences = []

    for i in range(rounds):

        qualy = fastf1.get_session(2023, i + 1, 'Q')
        qualy.load()
        circuits.append(qualy.event.Location.split('-')[0])
        drivers = list(np.unique(qualy.laps.pick_team(team)['Driver'].values))
        q1, q2, q3 = qualy.laps.split_qualifying_sessions()
        if drivers[0] in q3['Driver'].unique() and drivers[1] in q3['Driver'].unique():
            session = q3
        elif drivers[0] in q2['Driver'].unique() and drivers[1] in q2['Driver'].unique():
            session = q2
        else:
            session = q1
        try:
            d0_time = session.pick_driver(drivers[0]).pick_fastest()['LapTime'].total_seconds()
            d1_time = session.pick_driver(drivers[1]).pick_fastest()['LapTime'].total_seconds()

            if d0_time > d1_time:
                legend.append(f'{drivers[1]} faster')
                color.append('#0000FF')
            else:
                legend.append(f'{drivers[0]} faster')
                color.append('#FFA500')

            delta_diff = ((d0_time - d1_time) / d1_time) * 100
            differences.append(round(-delta_diff, 2))
        except AttributeError:
            print('No hay vuelta disponible')
            differences.append(np.nan)
            color.append('#0000FF')

    print(f'MEAN: {statistics.mean([i for i in differences if not np.isnan(i)])}')
    print(f'MEDIAN: {statistics.median([i for i in differences if not np.isnan(i)])}')

    fig, ax1 = plt.subplots(figsize=(7.2, 6.5), dpi=150)
    bars = plt.bar(circuits, differences, color=color)

    round_bars(bars, ax1, color)
    annotate_bars(bars, ax1, 0.01, 8, text_annotate='{height}%', ceil_values=False)

    legend_lines = []
    unique_colors = []
    unique_drivers = []
    i = 0
    for c_color in color:
        if c_color not in unique_colors:
            unique_colors.append(c_color)
            unique_drivers.append(legend[i])
            legend_p = Line2D([0], [0], color=c_color, lw=4)
            legend_lines.append(legend_p)
        i += 1

    plt.legend(legend_lines, unique_drivers,
               loc='lower left', fontsize='large')

    plt.axhline(0, color='white', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')
    plt.title(f'QUALY DIFFERENCE COMPARISON BETWEEN {team.upper()} TEAMMATES', font='Fira Sans', fontsize=2)
    plt.xticks(ticks=range(len(circuits)), labels=circuits,
               rotation=90, fontsize=12, fontname='Fira Sans')
    plt.xlabel('Circuit', font='Fira Sans', fontsize=16)
    plt.ylabel('Time diff (percentage)', font='Fira Sans', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'../PNGs/PACE DIFF BETWEEN {team} TEAMMATES.png', dpi=500)
    plt.show()
