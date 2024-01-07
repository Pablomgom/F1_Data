import statistics

import fastf1
import numpy as np
import pandas as pd
from fastf1.core import Laps
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt
from fastf1 import utils, plotting
from pyod.models.knn import KNN
from matplotlib.ticker import FuncFormatter
from timple.timedelta import strftimedelta
import matplotlib.patches as mpatches
from src.ergast_api.my_ergast import My_Ergast
from src.exceptions import qualy_by_year
from src.exceptions.custom_exceptions import QualyException
from src.plots.plots import rounded_top_rect, round_bars, annotate_bars
from src.utils.utils import append_duplicate_number
from src.variables.team_colors import team_colors_2023
from scipy import stats


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

    fig, ax = plt.subplots(figsize=(8, 8))
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
                print(f'{t}: {v}')
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

        for i, delta in enumerate(deltas):
            if np.isnan(delta):
                prev_index = max([j for j in range(i - 1, -1, -1) if not np.isnan(deltas[j])], default=None)
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


def percentage_qualy_ahead(start=2001, end=2024):
    ergast = My_Ergast()
    circuits = ['baku', 'monaco', 'jeddah', 'vegas', 'marina_bay', 'miami', 'valencia', 'villeneuve']
    qualy = ergast.get_qualy_results([i for i in range(start, end)]).content
    drivers_dict = {}
    for q in qualy:
        if len(q[q['circuitRef'].isin(circuits)]) > 0:
            drivers_in_q = q['fullName'].unique()
            for d in drivers_in_q:
                driver_data = q[q['fullName'] == d]
                driver_pos = min(driver_data['position'])
                driver_teams = driver_data['constructorName'].unique()
                for driver_team in driver_teams:
                    team_data = q[q['constructorName'] == driver_team]
                    team_data = team_data[team_data['fullName'] != d]
                    for teammate in team_data['fullName'].unique():
                        teammate_pos = min(q[q['fullName'] == teammate]['position'])
                        win = 1
                        if driver_pos > teammate_pos:
                            win = 0
                        if d not in drivers_dict:
                            drivers_dict[d] = [win]
                        else:
                            drivers_dict[d].append(win)
            print(f'{q["year"].loc[0]}: {q["circuitName"].loc[0]}')
    final_dict = {}
    h2h_dict = {}
    for d, w in drivers_dict.items():
        percentage = round((sum(w) / len(w)) * 100, 2)
        final_dict[d] = percentage
        h2h_dict[d] = f'({sum(w)}/{len(w)})'

    final_dict = dict(sorted(final_dict.items(), key=lambda item: item[1], reverse=True))
    for d, w in final_dict.items():
        print(f'{d}: {w}% {h2h_dict[d]}')


def qualy_diff_teammates(d1, start=1900, end=3000):
    def get_min_times(full_data, d1, d2):
        d1_time = min(full_data[full_data['fullName'] == d1]['q1'].loc[0],
                      full_data[full_data['fullName'] == d1]['q2'].loc[0]).total_seconds()
        d2_time = min(full_data[full_data['fullName'] == d2]['q1'].loc[0],
                      full_data[full_data['fullName'] == d2]['q2'].loc[0]).total_seconds()
        return d1_time, d2_time

    def process_times(year, q, d1_time, d2_time, total_laps_d1, total_laps_d2, delta_per_year, d1, d2):
        diff = round(d1_time - d2_time, 3)
        if abs(diff) < 5:
            display_time_comparison(year, q, d1_time, d2_time, diff, d1, d2)
            total_laps_d1.append(d1_time)
            total_laps_d2.append(d2_time)
            delta_per_year.setdefault((year, d2), []).append(diff)

    def process_sessions(year, q, full_data, d1, d2, sessions, total_laps_d1, total_laps_d2, delta_per_year):
        for s in sessions:
            s_data = full_data.dropna(subset=[s])
            if len(s_data) != 2:
                continue
            d1_time, d2_time = s_data[s_data['fullName'] == d1][s].iloc[0].total_seconds(), \
                s_data[s_data['fullName'] == d2][s].iloc[0].total_seconds()
            diff = round(d1_time - d2_time, 3)
            if abs(diff) < 5:
                display_time_comparison(year, q, d1_time, d2_time, diff, d1, d2, s)
                total_laps_d1.append(d1_time)
                total_laps_d2.append(d2_time)
                delta_per_year.setdefault((year, d2), []).append(diff)
                break
            else:
                print(f'No data for {year} {q["raceName"].iloc[0].replace("Grand Prix", "GP")} {s.upper()}')

    def display_time_comparison(year, q, d1_time, d2_time, diff, code_1, code_2, session=None):
        session_info = f" in {session.upper()}" if session else ""
        if d1_time > d2_time:
            faster_driver = code_2
            slowest_driver = code_1
        else:
            faster_driver = code_1
            slowest_driver = code_2

        print(
            f'{faster_driver} {abs(diff):.3f}s faster than {slowest_driver} in {year} {q["raceName"].iloc[0].replace("Grand Prix", "GP")}{session_info}')

    qualys = My_Ergast().get_qualy_results(list(range(start, end))).content
    delta_per_year = {}
    teammates_per_year = {}
    sessions = ['q3', 'q2', 'q1']
    total_laps_d1, total_laps_d2 = [], []

    for q in qualys:
        year = q['year'].iloc[0]
        race = q['round'].iloc[0]
        if race == 13:
            a = 1
        team_data = q[q['fullName'] == d1]
        if len(team_data) != 1:
            continue

        team = team_data['constructorName'].iloc[0]
        teammates = q[q['constructorName'] == team].query("fullName != @d1")['fullName'].values

        for d2 in teammates:
            teammates_per_year.setdefault(year, []).append(d2)
            full_data = q[q['fullName'].isin([d1, d2])]
            comparison_years = set(range(1984, 1996)) | {2003, 2004, 2005}

            if year in comparison_years:
                d1_time, d2_time = get_min_times(full_data, d1, d2)
                process_times(year, q, d1_time, d2_time, total_laps_d1, total_laps_d2, delta_per_year, d1, d2)
            else:
                process_sessions(year, q, full_data, d1, d2, sessions, total_laps_d1, total_laps_d2, delta_per_year)

    for y, d in delta_per_year.items():
        print(f'{y[0]} MEDIAN: {statistics.median(d):.3f}s '
              f'TRUNC MEAN: {stats.trim_mean(d, 0.1):.3f}s '
              f'{y[1]}')

    total_difference = []
    for l1, l2 in zip(total_laps_d1, total_laps_d2):
        delta_diff = (l1 - l2) / ((l1 + l2) / 2) * 100
        total_difference.append(delta_diff)

    median = statistics.median(total_difference)
    mean = statistics.mean(total_difference)
    print(f'MEDIAN DIFF {d1 if median < 0 else "TEAMMATE"} FASTER: {median:.3f}%')
    print(f'MEAN DIFF {d1 if mean < 0 else "TEAMMATE"} FASTER: {mean:.3f}%')


