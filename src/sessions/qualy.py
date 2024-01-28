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


def qualy_margin(circuit, start=1950, end=2050, order='Ascending'):
    """
       Prints the qualy margins in a given circuits

       Parameters:
       circuit(str): Circuit to analyze
       start (int, optional): Year of start. Default: 1950
       start (int, optional): Year of end. Default: 2024

    """

    qualy = My_Ergast().get_qualy_results([i for i in range(start, end)])
    sessions = ['q3', 'q2', 'q1']
    margins = {}
    for q in qualy.content:
        track = q['circuitRef'].loc[0]
        year = q['year'].loc[0]
        round = q['round'].loc[0]
        race_name = q['raceName'].loc[0]
        if track == circuit:
            q = q[q['Valid'] == True].reset_index(drop=True)
            p1 = q['fullName'].loc[0]
            p2 = q['fullName'].loc[1]
            if year > 2005 or (year == 2005 and round >= 7):
                for s in sessions:
                    p1_time = q[q['fullName'] == p1][s].loc[0]
                    p2_time = q[q['fullName'] == p2][s].loc[0]
                    if not pd.isna(p1_time) and not pd.isna(p2_time):
                        diff = (p2_time - p1_time).total_seconds()
                        margins[(year, race_name)] = (diff, f'{p1} {diff:.3f}s faster than {p2}')
                        break

            elif year == 2005 and round < 7:
                p1_time = q[q['fullName'] == p1]['q1'].loc[0] + q[q['fullName'] == p1]['q2'].loc[0]
                p2_time = q[q['fullName'] == p2]['q1'].loc[0] + q[q['fullName'] == p2]['q2'].loc[0]
                diff = (p2_time - p1_time).total_seconds()
                margins[(year, race_name)] = (diff, f'{p1} {diff:.3f}s faster than {p2}')
            else:
                final_times = []
                for d in [p1, p2]:
                    d_data = q[q['fullName'] == d]
                    times = []
                    for s_2 in sessions:
                        session_time = d_data[s_2].loc[0]
                        if not pd.isna(session_time):
                            times.append(d_data[s_2].loc[0])
                    final_times.append(min(times))
                diff_2 = (final_times[1] - final_times[0]).total_seconds()
                margins[(year, race_name)] = (diff_2, f'{p1} {diff_2:.3f}s faster than {p2}')

    if order == 'Ascending':
        margins = dict(sorted(margins.items(), key=lambda item: item[1][0]))
    else:
        margins = dict(sorted(margins.items(), key=lambda item: item[1][0], reverse=True))
    for r, m in margins.items():
        print(f'{r[0]}: {m[1]}')


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
    def get_min_value(full_data):
        times = []
        for i in sessions:
            s_time = full_data[i].loc[0]
            if not pd.isna(s_time):
                times.append(s_time)
        if len(times) == 0:
            return pd.NaT
        else:
            return min(times)

    def get_min_times(full_data, d1, d2):
        d1_time = get_min_value(full_data[full_data['fullName'] == d1]).total_seconds()
        d2_time = get_min_value(full_data[full_data['fullName'] == d2]).total_seconds()
        return d1_time, d2_time

    def process_times(year, q, full_data, d1_time, d2_time, total_laps_d1, total_laps_d2, delta_per_year, d1, d2):
        diff = round(d1_time - d2_time, 3)
        if abs(diff) < 1500:
            d1_pos = full_data[full_data['fullName'] == d1]['position'].loc[0]
            d2_pos = full_data[full_data['fullName'] == d2]['position'].loc[0]
            display_time_comparison(year, q, d1_time, d2_time, diff, d1, d2, d1_pos, d2_pos)
            total_laps_d1.append(d1_time)
            total_laps_d2.append(d2_time)
            delta_per_year.setdefault((year, d2), []).append(diff)
        else:
            print(f'No data for {year} {q["raceName"].iloc[0].replace("Grand Prix", "GP")} - {d2}')

    def process_sessions(year, q, full_data, d1, d2, sessions, total_laps_d1, total_laps_d2, delta_per_year):
        for s in sessions:
            s_data = full_data.dropna(subset=[s])
            if len(s_data) != 2:
                continue
            d1_time, d2_time = s_data[s_data['fullName'] == d1][s].iloc[0].total_seconds(), \
                s_data[s_data['fullName'] == d2][s].iloc[0].total_seconds()
            diff = round(d1_time - d2_time, 3)
            if abs(diff) < 5:
                positions = q.sort_values(by=s, ascending=True).reset_index(drop=True)
                d1_pos = positions[positions['fullName'] == d1].index[0] + 1
                d2_pos = positions[positions['fullName'] == d2].index[0] + 1
                display_time_comparison(year, q, d1_time, d2_time, diff, d1, d2, d1_pos, d2_pos, s)
                total_laps_d1.append(d1_time)
                total_laps_d2.append(d2_time)
                delta_per_year.setdefault((year, d2), []).append(diff)
                break
            else:
                print(f'No data for {year} {q["raceName"].iloc[0].replace("Grand Prix", "GP")} {s.upper()}')

    def display_time_comparison(year, q, d1_time, d2_time, diff, code_1, code_2, d1_pos, d2_pos, session=None):
        session_info = f" in {session.upper()}" if session else ""
        if d1_time > d2_time:
            faster_driver = code_2
            slowest_driver = code_1
            best_pos = d2_pos
            worst_pos = d1_pos
        else:
            faster_driver = code_1
            slowest_driver = code_2
            best_pos = d1_pos
            worst_pos = d2_pos

        print(
            f'{faster_driver} (P{best_pos}) {abs(diff):.3f}s faster than {slowest_driver} (P{worst_pos}) '
            f'in {year} {q["raceName"].iloc[0].replace("Grand Prix", "GP")}{session_info}')

    qualys = My_Ergast().get_qualy_results(list(range(start, end))).content
    delta_per_year = {}
    teammates_per_year = {}
    sessions = ['q3', 'q2', 'q1']
    total_laps_d1, total_laps_d2 = [], []

    for q in qualys:
        year = q['year'].iloc[0]
        race = q['round'].iloc[0]
        team_data = q[q['fullName'] == d1]
        if len(team_data) != 1:
            continue

        team = team_data['constructorName'].iloc[0]
        teammates = q[q['constructorName'] == team].query("fullName != @d1")['fullName'].values

        for d2 in teammates:
            teammates_per_year.setdefault(year, []).append(d2)
            full_data = q[q['fullName'].isin([d1, d2])]
            comparison_years = set(range(1950, 1996)) | {2003, 2004, 2005}

            if year in comparison_years:
                d1_time, d2_time = get_min_times(full_data, d1, d2)
                process_times(year, q, full_data, d1_time, d2_time, total_laps_d1, total_laps_d2, delta_per_year,
                              d1, d2)
            else:
                process_sessions(year, q, full_data, d1, d2, sessions, total_laps_d1, total_laps_d2, delta_per_year)

    for y, d in delta_per_year.items():
        trunc_mean = stats.trim_mean(d, 0.1)
        print(
            f'{"🔴" if trunc_mean > 0 else "🟢"}{y[0]}: {abs(trunc_mean):.3f}s {"faster" if trunc_mean < 0 else " slower"} than {y[1]}')

    for y, d in delta_per_year.items():
        mean = np.mean(d)
        median = np.median(d)
        print(f'{y[0]} MEAN: {mean:.3f}s {"faster" if mean < 0 else "slower"} MEDIAN: {median:.3f}s {"faster" if median < 0 else "slower"} ')

    total_difference = []
    for l1, l2 in zip(total_laps_d1, total_laps_d2):
        delta_diff = (l1 - l2) / ((l1 + l2) / 2) * 100
        total_difference.append(delta_diff)

    median = statistics.median(total_difference)
    mean = statistics.mean(total_difference)
    trunc_mean = stats.trim_mean(total_difference, 0.1)
    print(f'MEDIAN DIFF {d1 if median < 0 else "TEAMMATE"} FASTER: {median:.3f}%')
    print(f'MEAN DIFF {d1 if mean < 0 else "TEAMMATE"} FASTER: {mean:.3f}%')
    print(f'TRUNC DIFF {d1 if trunc_mean < 0 else "TEAMMATE"} FASTER: {trunc_mean:.3f}%')


def avg_qualy_pos_dif_per_years(start=2014, end=2024):
    qualys = My_Ergast().get_qualy_results([i for i in range(start, end)])
    drivers = []
    for q in qualys.content:
        drivers.extend(q['fullName'])
    drivers = set(drivers)
    delta_per_year = {}
    for d in drivers:
        for q in qualys.content:
            driver_data = q[q['fullName'] == d]
            if len(driver_data) > 0:
                valid = driver_data['Valid'].loc[0]
                if valid:
                    driver_pos = driver_data['position'].loc[0]
                    team = driver_data['constructorName'].loc[0]
                    teammates = q[(q['constructorName'] == team) & (q['fullName'] != d)]['fullName']
                    for t in teammates.values:
                        teammate_data = q[q['fullName'] == t]
                        teammate_valid = teammate_data['Valid'].loc[0]
                        year = q['year'].loc[0]
                        if teammate_valid:
                            teammate_pos = teammate_data['position'].loc[0]
                            if (year, d, t) not in delta_per_year:
                                delta_per_year[(year, d, t)] = [[], []]
                            delta_per_year[(year, d, t)][0].append(driver_pos)
                            delta_per_year[(year, d, t)][1].append(teammate_pos)

    print('---DRIVERS---')
    drivers_dict = dict(
        sorted(delta_per_year.items(), key=lambda item: np.mean(item[1][0]) - np.mean(item[1][1]), reverse=True))
    for t, l in drivers_dict.items():
        if (np.mean(l[0]) - np.mean(l[1])) < 0:
            print(f'{t[0]}: {t[1]} {abs((np.mean(l[0]) - np.mean(l[1]))):.2f} in front of {t[2]}')


def avg_qualy_pos_dif(driver):

    print(f'🚨{driver.upper()} AVERAGE QUALY POSITION DIFFERENCE AGAINST HIS TEAMMATES')

    qualys = My_Ergast().get_qualy_results([i for i in range(1950, 2100)])
    delta_per_year = {}
    for q in qualys.content:
        driver_data = q[q['fullName'] == driver]
        if len(driver_data) > 0:
            valid = driver_data['Valid'].loc[0]
            if valid:
                driver_pos = driver_data['position'].loc[0]
                team = driver_data['constructorName'].loc[0]
                teammates = q[(q['constructorName'] == team) & (q['fullName'] != driver)]['fullName']
                for t in teammates.values:
                    teammate_data = q[q['fullName'] == t]
                    teammate_valid = teammate_data['Valid'].loc[0]
                    year = q['year'].loc[0]
                    if teammate_valid:
                        teammate_pos = teammate_data['position'].loc[0]
                        if (year, t) not in delta_per_year:
                            delta_per_year[(year, t)] = [[], []]
                        delta_per_year[(year, t)][0].append(driver_pos)
                        delta_per_year[(year, t)][1].append(teammate_pos)
                    else:
                        print(f'{q["year"].loc[0]}: {q["raceName"].loc[0]} DSQ')
            else:
                print(f'{q["year"].loc[0]}: {q["raceName"].loc[0]} DSQ')

    for y, d in delta_per_year.items():
        driver_avg = np.median(d[0])
        teammate_avg = np.median(d[1])
        diff = driver_avg - teammate_avg
        diff_str = f'+{diff:.2f}' if diff > 0 else f'{diff:.2f}'
        print(f'{"🔴" if diff > 0 else "🟢" if diff < 0 else "🟰"}{y[0]}: {y[1]} ({diff_str})')

    print(f'If the value is less than 0, {driver} has a better average qualy position.')
    print(f'If the value is greater than 0, {driver} has a worst average qualy position.')
