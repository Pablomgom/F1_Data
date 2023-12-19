import re
import statistics
from collections import Counter

import fastf1
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt
from matplotlib import cm
from src.ergast_api.my_ergast import My_Ergast
from src.plots.plots import round_bars, annotate_bars
from src.variables.driver_colors import driver_colors_2023


def avg_driver_position(driver, team, year, session='Q'):
    """
        Get the avg qualy/race position

        Parameters:
        driver (str): A specific driver
        team (str): Team of the driver
        year (int): Year
        session (str): Q o R

   """

    ergast = Ergast()
    if session == 'Q':
        my_ergast = My_Ergast()
        data = my_ergast.get_qualy_results([year])
    else:
        data = ergast.get_race_results(season=year, limit=1000)

    position = []

    if driver is None:
        drivers = []
        for gp in data.content:
            for d in gp['driverCode']:
                if d not in ['LAW', 'DEV']:
                    drivers.append(d)
        drivers_array = set(drivers)
        drivers = {d: [] for d in drivers_array}
        prev_points = {d: [] for d in drivers_array}
        race_count = 0
        for gp in data.content:
            for d in drivers_array:
                gp_data = gp[gp['driverCode'] == d]
                if len(gp_data) > 0:
                    pos = gp_data['position'].values[0]
                    drivers[d].append(pos)
                    if race_count < len(data.content) - 1:
                        prev_points[d].append(pos)
            race_count += 1
        avg_grid = {}
        avg_grid_pre = {}
        for key, pos_array in drivers.items():
            mean_pos = round(np.mean(pos_array), 2)
            avg_grid[key] = mean_pos
        for key, pos_array in prev_points.items():
            mean_pos = round(np.mean(pos_array), 2)
            avg_grid_pre[key] = mean_pos
        difference = {key: avg_grid[key] - avg_grid_pre[key] for key in avg_grid}
        avg_grid = dict(sorted(avg_grid.items(), key=lambda item: item[1]))
        avg_grid_pre = dict(sorted(avg_grid_pre.items(), key=lambda item: item[1]))
        drivers = list(avg_grid.keys())
        avg_pos = list(avg_grid.values())
        colors = [driver_colors_2023.get(key, '#FFFFFF') for key in drivers]
        fig, ax = plt.subplots(figsize=(9, 7.2), dpi=150)  # Set the figure size (optional)
        bars = plt.bar(drivers, avg_pos, color=colors)  # Plot the bar chart with specific colors (optional)

        round_bars(bars, ax, colors, color_1=None, color_2=None, y_offset_rounded=-0.1, corner_radius=0.1)
        annotate_bars(bars, ax, 0.2, 10.5, text_annotate='default',
                      ceil_values=False, round=2)

        plt.xlabel('Drivers', font='Fira Sans', fontsize=14)  # x-axis label (optional)
        plt.ylabel('Avg Grid Position', font='Fira Sans', fontsize=14)  # y-axis label (optional)
        plt.title('Average Qualy Position Per Driver', font='Fira Sans', fontsize=20)  # Title (optional)
        plt.xticks(rotation=90, fontsize=11)
        plt.yticks(fontsize=11)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'../PNGs/Average grid position {year}.png', dpi=450)
        plt.show()
        pre_positions = list(avg_grid_pre.keys())
        for i in range(len(drivers)):
            driver = drivers[i]
            pre_pos = pre_positions.index(driver) + 1
            diff = round(difference[driver], 2)
            if diff > 0:
                diff = f'+{diff}'

            print(f'{i + 1}: {driver} - {avg_pos[i]} ({diff}) from {pre_pos}')
    else:
        for gp in data.content:
            session_data = gp[(gp['familyName'] == driver) & (gp['constructorRef'] == team)]
            if len(session_data) > 0:
                position.append(session_data['position'].values[0])
            else:
                print(f'{driver} not in {team}')

        print(np.round(np.mean(position), 2))
        return np.round(np.mean(position), 2), statistics.median(position)


def full_compare_drivers_season(year, d1, d2, team=None, mode='driver', split=None, d1_team=None, d2_team=None):
    """
       Compare a season for 2 drivers, with a lot of details

        Parameters:
        year (int): Data
        d1 (str): Driver 1
        d2 (str): Driver 2
        team (str, optional): Only for analyze a team. Default: None
        mode (str, optional): Changes between team and driver. Default: driver
        split (int, optional): To split a season for one team. Default: None
        d1_team (str, optional): Team of d1
        d2_team (str, optional): Team of d2
   """

    ergast = Ergast()
    if mode == 'team':
        race_results = ergast.get_race_results(season=year, constructor=team, limit=1000).content
        race_part_1 = [race_results[i] for i in range(split)]
        race_part_2 = [race_results[i] for i in range(split, len(race_results))]
        qualy_results = ergast.get_qualifying_results(season=year, constructor=team, limit=1000).content
        qualy_part_1 = [qualy_results[i] for i in range(split)]
        qualy_part_2 = [qualy_results[i] for i in range(split, len(qualy_results))]
        constructor_data_p1 = ergast.get_constructor_standings(season=year, constructor=team, round=split, limit=1000)
        constructor_data_p2 = ergast.get_constructor_standings(season=year, constructor=team, round=len(race_results),
                                                               limit=1000)

        avg_grid_p1 = np.round(np.mean([pos for i in qualy_part_1 for pos in i['position'].values]), 2)
        avg_grid_p2 = np.round(np.mean([pos for i in qualy_part_2 for pos in i['position'].values]), 2)

        positions_p1 = []
        positions_p2 = []
        dnf_p1 = []
        dnf_p2 = []

        for i in race_part_1:
            positions_df = i[i['status'].apply(lambda x: bool(re.search(r'(Finished|\+)', x)))]
            dnf_df = i[~i['status'].apply(lambda x: bool(re.search(r'(Finished|\+)', x)))]
            positions_p1.extend(positions_df['position'].values)
            dnf_p1.extend(dnf_df['position'].values)
        avg_pos_p1 = np.round(np.mean(positions_p1), 2)
        dnf_p1 = len(dnf_p1)

        for i in race_part_2:
            positions_df = i[i['status'].apply(lambda x: bool(re.search(r'(Finished|\+)', x)))]
            dnf_df = i[~i['status'].apply(lambda x: bool(re.search(r'(Finished|\+)', x)))]
            positions_p2.extend(positions_df['position'].values)
            dnf_p2.extend(dnf_df['position'].values)
        avg_pos_p2 = np.round(np.mean(positions_p2), 2)
        dnf_p2 = len(dnf_p2)

        top_10_d1 = len([pos for i in race_part_1 for pos in i['position'].values if pos <= 10])
        podium_d1 = len([pos for i in race_part_1 for pos in i['position'].values if pos <= 3])
        victories_d1 = len([pos for i in race_part_1 for pos in i['position'].values if pos == 1])
        top_10_d2 = len([pos for i in race_part_2 for pos in i['position'].values if pos <= 10])
        podium_d2 = len([pos for i in race_part_2 for pos in i['position'].values if pos <= 3])
        victories_d2 = len([pos for i in race_part_2 for pos in i['position'].values if pos == 1])
        points_d1 = constructor_data_p1.content[0]['points'].values[0]
        points_d2 = constructor_data_p2.content[0]['points'].values[0]
        team_pos_d1 = constructor_data_p1.content[0]['position'].values[0]
        team_pos_d2 = constructor_data_p2.content[0]['position'].values[0]

        print(f"""
            AVG.GRID: {avg_grid_p1} -> {avg_grid_p2}
            AVG.RACE: {avg_pos_p1} -> {avg_pos_p2}
            DNFs: {dnf_p1} -> {dnf_p2}
            TOP 10: {top_10_d1} -> {top_10_d2}
            PODIUMS: {podium_d1} -> {podium_d2}
            VICTORIES: {victories_d1} -> {victories_d2}
            POINTS: {points_d1} -> {points_d2 - points_d1}
            TEAM POS: {team_pos_d1} -> {team_pos_d2}

        """)

    else:
        if d1_team is not None:
            d1_avg_pos, median_d1_pos = avg_driver_position(d1, d1_team, year)
        else:
            d1_avg_pos, median_d1_pos = avg_driver_position(d1, team, year)
        if d2_team is not None:
            d2_avg_pos, median_d2_pos = avg_driver_position(d2, d2_team, year)
        else:
            d2_avg_pos, median_d2_pos = avg_driver_position(d2, team, year)
        my_ergast = My_Ergast()
        d1_race_results = my_ergast.get_race_results([year]).content
        for idx, race in enumerate(d1_race_results):
            d1_race_results[idx] = race[race['familyName'] == d1]
        d1_code = d1_race_results[0]['driverCode'].values[0]
        d1_mean_race_pos_no_dnf = np.mean([i['position'].values[0] for i in d1_race_results
                                           if re.search(r'(Finished|\+)', i['status'].max())])

        d1_dnf_count = len([i['position'].values[0] for i in d1_race_results
                            if not re.search(r'(Finished|\+)', i['status'].max())])

        d1_victories = len([i['position'].values[0] for i in d1_race_results if i['position'].values[0] == 1])
        d1_podiums = len([i['position'].values[0] for i in d1_race_results if i['position'].values[0] <= 3])
        d1_points = sum([i['points'].values[0] for i in d1_race_results])
        sprints_d1 = ergast.get_sprint_results(year, driver=d1).content
        points_sprint_d1 = sum([i['points'].values[0] for i in sprints_d1])
        d1_points += points_sprint_d1
        top_10_d1 = len([pos for i in d1_race_results for pos in i['position'].values if pos <= 10])

        d2_race_results = my_ergast.get_race_results([year]).content
        for idx, race in enumerate(d2_race_results):
            d2_race_results[idx] = race[race['familyName'] == d2]
        d2_code = d2_race_results[0]['driverCode'].values[0]
        d2_mean_race_pos_no_dnf = np.mean([i['position'].values[0] for i in d2_race_results
                                           if re.search(r'(Finished|\+)', i['status'].max())])

        d2_dnf_count = len([i['position'].values[0] for i in d2_race_results
                            if not re.search(r'(Finished|\+)', i['status'].max())])

        d2_victories = len([i['position'].values[0] for i in d2_race_results if i['position'].values[0] == 1])
        d2_podiums = len([i['position'].values[0] for i in d2_race_results if i['position'].values[0] <= 3])
        d2_points = sum([i['points'].values[0] for i in d2_race_results])
        sprints_d2 = ergast.get_sprint_results(year, driver=d2).content
        points_sprint_d2 = sum([i['points'].values[0] for i in sprints_d2])
        d2_points += points_sprint_d2
        top_10_d2 = len([pos for i in d2_race_results for pos in i['position'].values if pos <= 10])

        d1_percentage = round((d1_points / (d1_points + d2_points)) * 100, 2)
        d2_percentage = round((d2_points / (d1_points + d2_points)) * 100, 2)

        d1_laps_ahead = 0
        d2_laps_ahead = 0
        d1_percentage_ahead = 0
        d2_percentage_ahead = 0
        if year >= 2018:
            for i in range(len(d1_race_results)):
                session = fastf1.get_session(year, i + 1, 'R')
                session.load()
                d1_laps = session.laps.pick_driver(d1_code)
                d2_laps = session.laps.pick_driver(d2_code)
                laps_to_compare = min(len(d1_laps), len(d2_laps))
                d1_pos = d1_laps[:laps_to_compare]['Position']
                d2_pos = d2_laps[:laps_to_compare]['Position']

                for lap in range(laps_to_compare):
                    d1_lap_pos = d1_pos.iloc[lap]
                    d2_lap_pos = d2_pos.iloc[lap]
                    if d1_lap_pos < d2_lap_pos:
                        d1_laps_ahead += 1
                    else:
                        d2_laps_ahead += 1
            d1_percentage_ahead = round((d1_laps_ahead / (d1_laps_ahead + d2_laps_ahead)) * 100, 2)
            d2_percentage_ahead = round((d2_laps_ahead / (d1_laps_ahead + d2_laps_ahead)) * 100, 2)

        print(f"""
            AVG QUALY  POS: {d1} - {d1_avg_pos} --- {d2} - {d2_avg_pos}
            MEDIAN QUALY: {d1} - {median_d1_pos} --- {d2} - {median_d2_pos}
            AVG RACE POS NO DNF: {d1} - {d1_mean_race_pos_no_dnf} --- {d2} - {d2_mean_race_pos_no_dnf}
            VICTORIES: {d1} - {d1_victories} --- {d2} - {d2_victories}
            PODIUMS: {d1} - {d1_podiums} --- {d2} - {d2_podiums}
            TOP 10: {d1} - {top_10_d1} --- {d2} - {top_10_d2}
            DNFS: {d1} - {d1_dnf_count} --- {d2} - {d2_dnf_count}
            POINTS: {d1} - {d1_points} {d1_percentage}% --- {d2} - {d2_points} {d2_percentage}%
            LAPS IN FRONT: {d1} - {d1_laps_ahead} {d1_percentage_ahead}% --- {d2} - {d2_laps_ahead} {d2_percentage_ahead}%
        """)


def get_retirements_per_driver(driver, start=None, end=None):
    """
        Get retirements of a driver

        Parameters:
        driver (str): Driver
        start (int): Year of start. Default: 1950
        end (int): Year of end. Default: 2024

   """

    ergast = Ergast()
    positions = pd.Series(dtype=object)

    for i in range(start, end):
        races = ergast.get_race_results(season=i, limit=1000)
        total_races = races.content
        for race in total_races:
            race = race[race['familyName'] == driver]
            if not pd.isna(race['status'].max()):
                if re.search(r'(Spun off|Accident|Collision|Damage)', race['status'].max()):
                    positions = pd.concat([positions, pd.Series(['Crash'])], ignore_index=True)
                elif re.search(r'(Finished|\+)', race['status'].max()):
                    positions = pd.concat([positions, pd.Series(['P' + str(race['position'].max())])],
                                          ignore_index=True)

                    if race['position'].max() == 18:
                        print(race)

                else:
                    positions = pd.concat([positions, pd.Series(['Mech DNF'])], ignore_index=True)
        print(i)

    positions = positions.value_counts()
    positions = pd.DataFrame({'Category': positions.index, 'Count': positions.values})
    positions['Sort_Key'] = positions['Category'].replace({'Mech DNF': 'P100', 'Crash': 'P101'})
    positions['Sort_Key'] = positions['Sort_Key'].str.extract('(\d+)').astype(int)
    positions = positions.sort_values(by=['Count', 'Sort_Key'], ascending=[False, True]).drop('Sort_Key', axis=1)
    positions.reset_index(drop=True)
    positions = pd.Series(positions['Count'].values, index=positions['Category'])
    printed_data = pd.DataFrame(positions).reset_index()
    printed_data.columns = ['Position', 'Times']
    printed_data['Percentage'] = round((printed_data['Times'] / sum(printed_data['Times'])) * 100, 2)
    N = 8
    top_N = positions.nlargest(N)
    top_N['Other'] = positions.iloc[N:].sum()

    fig, ax = plt.subplots(figsize=(7.2, 6.5), dpi=150)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    colormap = cm.get_cmap('tab20', len(top_N))
    colors = [colormap(i) for i in range(len(top_N))]

    def autopct_generator(values):
        labels_plotted = []

        def inner_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            for i, (value, label) in enumerate(zip(values, labels)):
                if val == value and label not in labels_plotted:
                    labels_plotted.append(label)
                    return "{p:.2f}%\n({v:d})\n{label}".format(p=pct, v=val, label=label)
            return ""

        return inner_autopct

    # Generate labels and corresponding values
    labels = top_N.index.tolist()
    values = top_N.values

    # Adjust pie plot call
    top_N.plot.pie(ax=ax, autopct=autopct_generator(values),
                   labels=['' for _ in labels], legend=False,
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, colors=colors,
                   textprops={"color": "black", "ha": "center"})

    plt.title(f'{driver} historical positions (Total races: {positions.sum()})',
              font='Fira Sans', fontsize=16, color='white')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver} finish history', dpi=450)
    plt.show()
    print(printed_data)


def compare_drivers_season(d_1, d_2, season, DNFs=False):
    """
        Compare a season for 2 drivers

        Parameters:
        d_1 (str): Driver 1
        d_2 (str): Driver 2
        DNFs (bool): Count DNFs

   """

    ergast = Ergast()

    races = ergast.get_race_results(season=season, limit=1000)
    sprints = ergast.get_sprint_results(season=season, limit=1000)
    qualys = ergast.get_qualifying_results(season=season, limit=1000)
    total_races = races.content

    race_result = []
    qualy_result = []
    d1_points = 0
    d2_points = 0

    def process_drivers(array, race_type, d1_points, d2_points):
        for race in array:
            best_pos = race[race['familyName'].isin([d_1, d_2])]['position'].min()
            status = race[race['familyName'].isin([d_1, d_2])]['status'].values
            status = [i for i in status if '+' in i or 'Finished' in i]
            driver = race[race['position'] == best_pos]['driverCode'].min()
            if (DNFs and len(status) == 2) or not DNFs:
                race_result.append(driver + f' - {race_type}')
                d1_points += race[race['familyName'] == d_1]['points'][0]
                d2_points += race[race['familyName'] == d_2]['points'][0]


        return d1_points, d2_points

    d1_points, d2_points = process_drivers(total_races, 'Race', d1_points, d2_points)
    d1_points, d2_points = process_drivers(sprints.content, 'Sprint', d1_points, d2_points)

    for qualy in qualys.content:
        best_pos = qualy[qualy['familyName'].isin([d_1, d_2])]['position'].min()
        if len(best_pos) == 2:
            driver = qualy[qualy['position'] == best_pos]['driverCode'].min()
            qualy_result.append(driver)

    print(Counter(race_result))
    print(f'QUALYS: {Counter(qualy_result)}')
    print(f'{d_1} points: {d1_points}')
    print(f'{d_2} points: {d2_points}')


def get_driver_results_circuit(driver, circuit, start=None, end=None):
    """
        Get the driver results on a circuit

        Parameters:
        driver (str): A specific driver
        circuit (int): A specific circuit
        start (int): Year of start
        end (int): Year of end
   """

    if start is None:
        start = 1950
    if end is None:
        end = 2024

    ergast = Ergast()
    for year in range(start, end):
        round_number = ergast.get_race_schedule(season=year, circuit=circuit, limit=1000)
        if len(round_number):
            round_number = round_number.values[0][1]
            results = ergast.get_race_results(season=year, round=round_number, limit=1000)
            if len(results.content) > 0:
                results = results.content[0]
                results = results[results['driverId'] == driver]
                if len(results) > 0:
                    grid = results['grid'].values[0]
                    position = results['position'].values[0]
                    status = results['status'].values[0]
                    if '+' not in status and 'Finished' not in status:
                        position = 'DNF'
                    else:
                        position = 'P' + str(position)
                    team = results['constructorName'].values[0]
                    print(f'{year}: From P{grid} to {position} with {team}')
