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
from src.utils.utils import get_medal
from src.variables.driver_colors import driver_colors_2023, driver_colors


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
        race_count = 0
        for gp in data.content:
            for d in drivers_array:
                gp_data = gp[gp['driverCode'] == d]
                if len(gp_data) > 0:
                    pos = gp_data['position'].values[0]
                    drivers[d].append(pos)
            race_count += 1
        avg_grid = {}
        for key, pos_array in drivers.items():
            mean_pos = round(np.mean(pos_array), 2)
            avg_grid[key] = mean_pos

        avg_grid = dict(sorted(avg_grid.items(), key=lambda item: item[1]))
        drivers = list(avg_grid.keys())
        avg_pos = list(avg_grid.values())
        colors = [driver_colors.get(year).get(key, '#FFFFFF') for key in drivers]
        fig, ax = plt.subplots(figsize=(9, 7.2), dpi=150)  # Set the figure size (optional)
        bars = plt.bar(drivers, avg_pos, color=colors)  # Plot the bar chart with specific colors (optional)

        round_bars(bars, ax, colors, color_1=None, color_2=None, y_offset_rounded=-0.1, corner_radius=0.1)
        annotate_bars(bars, ax, 0.2, 10.5, text_annotate='default',
                      ceil_values=False, round=2)
        plt.title('Average Qualy Position Per Driver', font='Fira Sans', fontsize=24)
        plt.xticks(rotation=90, fontsize=14)
        plt.yticks(fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'../PNGs/Average grid position {year}.png', dpi=450)
        plt.show()
        for i in range(len(drivers)):
            driver = drivers[i]
            print(f'{i + 1}: {driver} - {avg_pos[i]:.2f}')
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


def get_retirements_per_driver(driver, start=1950, end=2100):
    """
        Get retirements of a driver

        Parameters:
        driver (str): Driver
        start (int): Year of start. Default: 1950
        end (int): Year of end. Default: 2024

   """

    ergast = My_Ergast()
    races = ergast.get_race_results([i for i in range(start, end)]).content
    positions = pd.Series(dtype=object)

    for race in races:
        race = race[race['fullName'] == driver]
        if not pd.isna(race['status'].max()):
            if re.search(r'(Spun off|Accident|Collision|Damage|damage)', race['status'].max()):
                positions = pd.concat([positions, pd.Series(['Crash'])], ignore_index=True)
            elif re.search(r'(Finished|\+)', race['status'].max()):
                positions = pd.concat([positions, pd.Series(['P' + str(race['position'].max())])],
                                      ignore_index=True)
            else:
                positions = pd.concat([positions, pd.Series(['Mech DNF'])], ignore_index=True)

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

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

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

    labels = top_N.index.tolist()
    values = top_N.values

    wedges, texts, autotexts = ax.pie(values, autopct=autopct_generator(values),
                                      labels=['' for _ in labels],
                                      wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, colors=colors,
                                      textprops={"color": "black", "ha": "center"},
                                      radius=1.25)

    for autotext in autotexts:
        autotext.set_fontsize(12)

    plt.title(f'{driver} historical positions (Total races: {positions.sum()})',
              font='Fira Sans', fontsize=16, color='white')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver} finish history', dpi=450)
    plt.show()
    printed_data['Rank'] = printed_data['Percentage'].rank(method='min', ascending=False)
    for index, row in printed_data.iterrows():
        position = row['Position']
        times = row['Times']
        percentage = row['Percentage']
        rank = int(row['Rank'])
        print(f'{rank} - {position}: {percentage}% ({times}/{positions.sum()})')


def race_qualy_h2h(d_1, start=1950, end=2100):
    """
        Compare a season for 2 drivers

        Parameters:
        d_1 (str): Driver 1
        d_2 (str): Driver 2
        DNFs (bool): Count DNFs

   """

    ergast = My_Ergast()
    years = [i for i in range(start, end)]
    races = ergast.get_race_results(years)
    qualys = ergast.get_qualy_results(years).content
    total_races = races.content

    race_result = {}
    qualy_result = {}
    d1_points = {}
    d2_points = {}
    race_h2h = pd.DataFrame(columns=['Driver', 'Rival', 'Team', 'Year', 'Round', 'Result', 'Reason'])
    qualy_h2h = pd.DataFrame(columns=['Driver', 'Rival', 'Team', 'Year', 'Round', 'Result', 'Reason'])

    def process_drivers(array, race_type, d1_points, d2_points, dataframe):
        for race in array:
            df_data = {'Driver': d_1}
            year = race['year'].loc[0]
            round_id = race['round'].loc[0]
            if len(race[race['fullName'] == d_1]) == 1:
                team = race[race['fullName'] == d_1]['constructorName'].loc[0]
                teammates = race[(race['constructorName'] == team) & (race['fullName'] != d_1)]['fullName'].values
                for d_2 in teammates:
                    if len(race[race['fullName'] == d_2]) == 1:
                        status = []
                        valid = []
                        if race_type == 'Race':
                            status = race[race['fullName'].isin([d_1, d_2])]['status'].values
                            status = [i for i in status if '+' in i or 'Finished' in i]
                        else:
                            valid = race[race['fullName'].isin([d_1, d_2])]['Valid'].values
                            valid = [i for i in valid if i]
                        d1_pos = race[race['fullName'] == d_1]['position'].loc[0]
                        d2_pos = race[race['fullName'] == d_2]['position'].loc[0]
                        driver = None
                        if d1_pos < d2_pos:
                            driver = d_1
                        elif d1_pos > d2_pos:
                            driver = d_2
                        if len(status) == 2 or len(valid) == 2:
                            if race_type == 'Race':
                                if year not in race_result:
                                    race_result[year] = {}
                                    d1_points[year] = {}
                                    d2_points[year] = {}
                                if d_2 not in race_result[year]:
                                    race_result[year][d_2] = []
                                    d1_points[year][d_2] = 0
                                    d2_points[year][d_2] = 0
                                if driver is not None:
                                    race_result[year][d_2].append(driver)
                                d1_points[year][d_2] += race[race['fullName'] == d_1]['points'][0]
                                d2_points[year][d_2] += race[race['fullName'] == d_2]['points'][0]
                            else:
                                if year not in qualy_result:
                                    qualy_result[year] = {}
                                if d_2 not in qualy_result[year]:
                                    qualy_result[year][d_2] = []
                                if driver is not None:
                                    qualy_result[year][d_2].append(driver)

                            df_data['Driver'] = d_1
                            df_data['Rival'] = d_2
                            df_data['Team'] = team
                            df_data['Year'] = [year]
                            df_data['Round'] = [round_id]
                            df_data['Result'] = [1 if driver == d_1 else (0 if driver == d_2 else None)]
                            df_data['Reason'] = np.nan
                            df_to_append = pd.DataFrame(df_data)
                            dataframe = dataframe._append(df_to_append)
                            df_data = {}
                        else:
                            df_data['Driver'] = d_1
                            df_data['Rival'] = d_2
                            df_data['Team'] = team
                            df_data['Year'] = [year]
                            df_data['Round'] = round_id
                            df_data['Result'] = np.nan
                            df_data[
                                'Reason'] = f"At least one of the drivers {'did not finish' if race_type == 'Race' else 'was DSQ'}"
                            df_to_append = pd.DataFrame(df_data)
                            dataframe = dataframe._append(df_to_append)
                            df_data = {}
                    else:
                        df_data['Driver'] = d_1
                        df_data['Rival'] = d_2
                        df_data['Team'] = team
                        df_data['Year'] = [year]
                        df_data['Round'] = round_id
                        df_data['Result'] = np.nan
                        df_data['Reason'] = f'{d_2} drove multiple cars'
                        df_to_append = pd.DataFrame(df_data)
                        dataframe = dataframe._append(df_to_append)
                        df_data = {}
            elif len(race[race['fullName'] == d_1]) > 1:
                df_data['Driver'] = d_1
                df_data['Rival'] = np.nan
                df_data['Team'] = np.nan
                df_data['Year'] = [year]
                df_data['Round'] = [round_id]
                df_data['Result'] = np.nan
                df_data['Reason'] = f'{d_1} drove multiple cars'
                df_to_append = pd.DataFrame(df_data)
                dataframe = dataframe._append(df_to_append)
                df_data = {}
        return d1_points, d2_points, dataframe

    d1_points, d2_points, race_h2h = process_drivers(total_races, 'Race', d1_points, d2_points, race_h2h)
    _, _, qualy_h2h = process_drivers(qualys, 'Qualy', d1_points, d2_points, qualy_h2h)

    def print_results(data, type):
        driver_ahead = 0
        teammate_ahead = 0
        for year, results in data.items():
            printed_line = ''
            for teammate, h2h in results.items():
                count = Counter(h2h)
                printed_line += f'{year}:'
                d_1_curr_year = 0
                teammate_curr_year = 0
                for name, times in count.items():
                    if name == d_1:
                        driver_ahead += times
                        d_1_curr_year = times
                    else:
                        teammate_ahead += times
                        teammate_curr_year = times
                printed_line += f' {d_1.split(" ", 1)[1]} {d_1_curr_year} - {teammate_curr_year} {teammate.split(" ", 1)[1]}'
                dot = '🟰'
                if d_1_curr_year < teammate_curr_year:
                    dot = '🔴'
                elif d_1_curr_year > teammate_curr_year:
                    dot = '🟢'
                printed_line = f'{dot}{printed_line}'
                print(printed_line)
                printed_line = ''
        try:
            print(
                f'{type.upper()} H2H: {driver_ahead / (driver_ahead + teammate_ahead) * 100:.2f}% ({driver_ahead}/{driver_ahead + teammate_ahead})')
        except:
            print(f'No {type} data for {d_1}')

    print_results(qualy_result, 'qualy')
    print_results(race_result, 'race')

    def update_csv(path, new_data):
        existing_df = pd.read_csv(path)
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        final_df = combined_df.drop_duplicates(subset=['Driver', 'Year', 'Round'], keep='first')
        final_df.to_csv(path, index=False)
        print(f'New data stored in {path}: {len(final_df) - len(existing_df)}')

    update_csv('../resources/csv/race_h2h.csv', race_h2h)
    update_csv('../resources/csv/qualy_h2h.csv', qualy_h2h)


def update_results(year, round):
    drivers = set(My_Ergast().get_race_results([year], round).content[0]['fullName'].values)
    for d in drivers:
        race_qualy_h2h(d, start=year)


def get_driver_results_circuit(driver, circuit, start=1950, end=2100):
    """
        Get the driver results on a circuit

        Parameters:
        driver (str): A specific driver
        circuit (int): A specific circuit
        start (int): Year of start
        end (int): Year of end
   """

    races = My_Ergast().get_race_results([i for i in range(start, end)])
    print(f'🚨{driver.upper()} RESULTS IN {circuit.upper()}\n')
    for r in races.content:
        results = r[(r['fullName'] == driver) & (r['circuitRef'] == circuit)]
        if len(results) > 0:
            year = r['year'].loc[0]
            grid = results['grid'].values[0]
            position = results['position'].values[0]
            status = results['status'].values[0]
            if status == 'Disqualified':
                position = 'DSQ'
            elif '+' not in status and 'Finished' not in status:
                position = 'DNF'
            else:
                position = 'P' + str(position)
            team = results['constructorName'].values[0]
            medal = get_medal(position)
            print(f'{medal}{year}: From P{grid} to {position} with {team}')


def driver_results_per_year(driver, start=1900, end=2100):
    races = My_Ergast().get_race_results([i for i in range(start, end)])
    positions_gained = 0
    for r in races.content:
        driver_data = r[r['fullName'] == driver]
        race_name = driver_data['raceName'].loc[0].replace('Grand Prix', 'GP')
        grid_pos = driver_data['grid'].loc[0]
        grid_integer = grid_pos
        grid_pos = f'P{grid_pos}'
        result = driver_data["position"].loc[0]
        status = driver_data['status'].loc[0]
        if status != 'Finished' and '+' not in status:
            result = 'DNF'
        if result != 'DNF':
            positions_gained += grid_integer - result
            result = f'P{result}'
        medal = get_medal(result)
        print(f'{medal}From {grid_pos} to {result} in the {race_name}')
    print(f'\nPositions gained (excluding DNFs): {positions_gained}')


def qualy_race_streaks(driver, session='qualy', win=True):
    races = pd.read_csv('../resources/ergast_data/races.csv')
    if session == 'qualy':
        all_qualys = pd.read_csv('../resources/csv/qualy_h2h.csv')
    else:
        all_qualys = pd.read_csv('../resources/csv/race_h2h.csv')
    all_qualys = all_qualys[all_qualys['Driver'] == driver]
    qualys = all_qualys[~pd.isna(all_qualys['Result'])]
    qualys = qualys.reset_index(drop=True)
    qualys['Round_diff'] = qualys.index + 1
    qualys['diff'] = qualys['Result'].diff()
    qualys.loc[(qualys['Round_diff'] == 1) & (qualys['Result'] == 1), 'diff'] = 1
    qualys.loc[(qualys['Round_diff'] == 1) & (qualys['Result'] == 0), 'diff'] = 0
    qualys = qualys[qualys['diff'].isin([-1, 1])]
    qualys['Finish_round'] = qualys['Round'].shift(-1)
    qualys['Finish_year'] = qualys['Year'].shift(-1)
    qualys['Streak number'] = qualys['Round_diff'].shift(-1) - qualys['Round_diff']
    qualys = qualys[qualys['diff'] == (1 if win else -1)]
    valid_sessions = all_qualys[~pd.isna(all_qualys['Result'])]
    qualys.loc[pd.isna(qualys['Streak number']), 'Streak number'] = len(valid_sessions) - qualys['Round_diff'] + 1
    qualys = qualys[~pd.isna(qualys['Streak number'])].sort_values(by=['Streak number', 'Year', 'Round'],
                                                                   ascending=[False, True, True])
    for index, row in qualys.iterrows():
        year_start = row['Year']
        round_start = row['Round']
        streak_number = int(row['Streak number'])
        try:
            year_finish = int(row['Finish_year'])
            round_finish = int(row['Finish_round']) - 1
        except:
            year_finish = all_qualys['Year'].max()
            round_finish = all_qualys[all_qualys['Year'] == year_finish]['Round'].max()

        if round_finish == 0:
            prev_year_data = (valid_sessions[valid_sessions['Year'] < year_finish]
                              .sort_values(by=['Year', 'Round'], ascending=[False, False]))
            year_finish = prev_year_data['Year'].loc[0]
            round_finish = prev_year_data['Round'].loc[0]

        start_race = races[(races['year'] == year_start) & (races['round'] == round_start)]['raceName'].loc[0]
        if streak_number != 1:
            if win:
                finish_race = races[(races['year'] == year_finish) & (races['round'] == round_finish)]['raceName'].loc[
                    0]
            else:
                finish_race = (races[(races['year'] <= year_finish) &
                                     ~((races['year'] == year_finish) & (races['round'] > round_finish))]
                .sort_values(by=['year', 'round'], ascending=[False, False]).reset_index(drop=True)['raceName'].loc[0])
            print(f'{streak_number}: From {year_start} {start_race.replace("Grand Prix", "GP")} '
                  f'to {year_finish} {finish_race.replace("Grand Prix", "GP")}')
        else:
            print(f'{streak_number}: {year_start} {start_race.replace("Grand Prix", "GP")}')
