import re
from collections import defaultdict, Counter
from pySankey import sankey
import pandas as pd
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from timple.timedelta import strftimedelta

from src.ergast_api.my_ergast import My_Ergast
from src.plots.table import render_mpl_table
from src.variables.team_colors import team_colors


def qualy_results_ergast(year, round):
    """
        Plot the qualy result of a GP with Ergast API

        Parameters:
        year (int): Year
        round (int): Round of GP

    """

    ergast = Ergast()
    qualy = ergast.get_qualifying_results(season=year, round=round, limit=1000)

    n_drivers = len(qualy.content[0]['Q1'])

    n_drivers_session = int((n_drivers - 10) / 2)
    qualy_times = []
    pole_lap = []
    colors = []
    for i in range(n_drivers):
        if i < 10:
            qualy_times.append(qualy.content[0]['Q3'][i])
        elif i >= 10 and i < (10 + n_drivers_session):
            qualy_times.append(qualy.content[0]['Q2'][i])
        elif i >= (10 + n_drivers_session):
            qualy_times.append(qualy.content[0]['Q1'][i])

        pole_lap.append(qualy.content[0]['Q3'][0])
        colors.append(team_colors[qualy.content[0]['constructorName'][i]])

    delta_time = pd.Series(qualy_times) - pd.Series(pole_lap)
    delta_time = delta_time.fillna(pd.Timedelta(days=0))

    fig, ax = plt.subplots()
    ax.barh(qualy.content[0]['driverCode'], delta_time, color=colors, edgecolor='grey')

    ax.set_yticks(qualy.content[0]['driverCode'])
    ax.set_yticklabels(qualy.content[0]['driverCode'])

    # show fastest at the top
    ax.invert_yaxis()
    ax.axhline(9.5, color='black', linestyle='-', linewidth=1)
    ax.text(max(delta_time).total_seconds() * 1e9, 11, 'Q2', va='bottom',
            ha='right', fontsize=14)
    ax.text(max(delta_time).total_seconds() * 1e9, 1, 'Q3', va='bottom',
            ha='right', fontsize=14)
    ax.text(max(delta_time).total_seconds() * 1e9, 16, 'Q1', va='bottom',
            ha='right', fontsize=14)
    # Horizontal bar at index 16
    ax.axhline(n_drivers_session + 9.5, color='black', linestyle='-', linewidth=1)

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

    lap_time_string = strftimedelta(pole_lap[0], '%m:%s.%ms')

    plt.suptitle(f"{qualy.description['raceName'][0]} {qualy.description['season'].min()} Qualifying\n"
                 f"Fastest Lap: {lap_time_string} ({qualy.content[0]['driverCode'][0]})")

    def custom_formatter(x, pos):
        return round(x * 100000, 1)

    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    plt.xlabel("Seconds")
    plt.ylabel("Driver")

    plt.savefig(f"../PNGs/{qualy.description['raceName'][0]} {qualy.description['season'].min()} Qualifying", dpi=400)
    plt.show()


def get_position_changes(year, round):
    """
        Plot the results of a race, with the position changes, in a table

        Parameters:
        year (int): Year
        round (int): Round of GP

    """

    race = Ergast().get_race_results(season=year, round=round, limit=1000)

    finish = race.content[0][['familyName', 'givenName', 'grid', 'status', 'constructorName']]
    finish['Driver'] = finish['givenName'] + ' ' + finish['familyName']
    finish['Finish'] = range(1, finish.shape[0] + 1)
    finish.loc[(finish['grid'] == 5) & (finish['Driver'] == 'Guanyu Zhou'), 'grid'] = 15
    finish['grid'].replace(0, 20, inplace=True)
    finish.loc[finish['status'].isin(['Did not qualify', 'Did not prequalify']), 'grid'] = finish['Finish']
    finish['Grid change'] = finish['grid'] - finish['Finish']
    # finish['grid'].replace(20, 'Pit Lane', inplace=True)
    finish['Team'] = finish['constructorName']

    race_diff_times = []
    for race_content in race.content:
        for i in range(len(race_content['totalRaceTime'])):
            if i == 0:
                race_time = race_content['totalRaceTime'][i]
                hours = race_time.seconds // 3600
                minutes = ((race_time.seconds // 60) % 60)
                seconds = race_time.seconds % 60
                milliseconds = race_time.microseconds // 1000
                race_time = f"{hours}:{str(minutes).ljust(2, '0')}:{str(seconds).ljust(3, '0')}" \
                            f".{str(milliseconds).ljust(3, '0')}"
                race_diff_times.append(race_time)
            else:
                race_time = race_content['totalRaceTime'][i]
                if pd.isna(race_time):
                    race_diff_times.append(None)
                else:
                    minutes = (race_time.seconds // 60) % 60
                    seconds = race_time.seconds % 60
                    milliseconds = race_time.microseconds // 1000
                    race_time = f"+{str(minutes).zfill(1)}:{str(seconds).zfill(2)}.{(str(milliseconds).zfill(2)).rjust(3, '0')}"
                    race_diff_times.append(race_time)

    finish['status'] = pd.Series(race_diff_times).combine_first(finish['status'])

    def modify_grid_change(value):
        if value > 0:
            return '+' + str(value)
        elif value == 0:
            return 'Equal'
        else:
            return str(value)

    finish['Grid change'] = finish['Grid change'].apply(modify_grid_change)
    finish.rename(columns={'status': 'Status', 'grid': 'Starting position'}, inplace=True)

    finish = finish[['Finish', 'Driver', 'Team', 'Starting position', 'Status', 'Grid change']]

    render_mpl_table(finish)
    plt.title(f"Race Results - {race.description['raceName'].min()} - {race.description['season'].min()}",
              font='Fira Sans', fontsize=40)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../PNGs/Race Results - {race.description['raceName'].min()} - {race.description['season'].min()}",
                bbox_inches='tight', dpi=400)
    plt.show()


def compare_my_ergast_teammates(driver, start=2001, end=2024):
    """
    Compare a driver against his teammates
    :param given: Name
    :param family: Surname
    :param start: Year of start
    :param end: Year of end
    :return: None
    """

    def process_data(session, d_data, t_data, col, race_data):
        driver_data = session[session['fullName'] == driver]
        if len(driver_data) == 1:
            team = driver_data['constructorName'].values[0]
            team_data = session[session['constructorName'] == team]
            team_data = team_data[team_data['fullName'] != driver]
            if len(team_data) == 1:
                d_position = driver_data[col].values[0]
                t_position = team_data[col].values[0]
                if d_position == 0:
                    d_position = 200
                if t_position == 0:
                    t_position = 200

                if d_position < t_position:
                    d_data[0] += 1
                else:
                    t_data[0] += 1
                driver_race_data = race_data[race_data['fullName'] == driver]
                team_race_data = race_data[race_data['constructorName'] == team]
                team_race_data = team_race_data[team_race_data['fullName'] != driver]
                d_grid = driver_race_data['grid'].values[0]
                t_grid = team_race_data['grid'].values[0]
                if d_grid == 1:
                    d_data[1] += 1
                    # print(f'{d_position} - {driver_race_data["year"].min()} - {driver_race_data["raceName"].min()}')
                elif t_grid == 1:
                    t_data[1] += 1
                    # print(f'{t_position} - {team_data["year"].min()} - {team_data["raceName"].min()}')

    my_ergast = My_Ergast()
    q = my_ergast.get_qualy_results([i for i in range(start, end)])
    r = my_ergast.get_race_results([i for i in range(start, end)])
    d_data = [0, 0, 0, 0, 0, 0, 0, 0]
    t_data = [0, 0, 0, 0, 0, 0, 0, 0]

    index = 0
    for qualy in q.content:
        process_data(qualy, d_data, t_data, 'position', r.content[index])
        index += 1

    for race in r.content:
        driver_data = race[race['fullName'] == driver]
        if len(driver_data) == 1:
            team = driver_data['constructorName'].values[0]
            team_data = race[race['constructorName'] == team]
            team_data = team_data[team_data['fullName'] != driver]
            d_points = driver_data['points'].values[0]
            d_data[7] += d_points
            if len(team_data) == 1:
                d_status = driver_data['status'].values[0]
                t_status = team_data['status'].values[0]
                d_position = driver_data['position'].values[0]
                t_position = team_data['position'].values[0]
                d_points = driver_data['points'].values[0]
                t_points = team_data['points'].values[0]
                # VICTORIES
                if d_position == 1:
                    d_data[3] += 1
                if t_position == 1:
                    t_data[3] += 1
                # PODIUMS
                if d_position in [1, 2, 3]:
                    d_data[4] += 1
                if t_position in [1, 2, 3]:
                    t_data[4] += 1
                # POINT FINISHES
                if d_points > 0:
                    d_data[5] += 1
                if t_points > 0:
                    t_data[5] += 1
                # TOTAL POINTS
                t_data[7] += t_points
                if re.search(r'(Finished|\+)', d_status) and re.search(r'(Finished|\+)', t_status):
                    if d_position < t_position:
                        d_data[2] += 1
                    else:
                        t_data[2] += 1
                    print(f'{d_status} - {driver_data["year"].min()} - {driver_data["raceName"].min()}')

                else:
                    if not re.search(r'(Finished|\+)', d_status):
                        # print(f'{d_status} - {driver_data["year"].min()} - {driver_data["raceName"].min()}')
                        d_data[6] += 1
                    if not re.search(r'(Finished|\+)', t_status):
                        t_data[6] += 1
                        # print(f'{t_status} - {driver_data["year"].min()} - {driver_data["raceName"].min()}')

    print(d_data, t_data)


def get_driver_laps(year):
    """
    Get the percetange of laps completed by driver per year
    :param year: Year of analysis
    :return: None
    """

    ergast = My_Ergast()
    races = ergast.get_race_results([year])
    drivers = []
    for r in races.content:
        d_race = r['fullName'].values
        for d in d_race:
            drivers.append(d)

    drivers = set(drivers)
    drivers_dict = {}
    for d in drivers:
        drivers_dict[d] = [[], []]

    for r in races.content:
        max_laps = r['laps'].max()
        d_race = r['fullName'].values
        for d in d_race:
            d_data = r[r['fullName'] == d]
            current_data = drivers_dict[d]
            current_data[0].append(max_laps)
            current_data[1].append(d_data['laps'].values[0])

    laps_dict = {}
    for driver, laps in drivers_dict.items():
        total_laps = sum(laps[0])
        completed_laps = sum(laps[1])
        percentage = round((completed_laps / total_laps) * 100, 2)
        laps_dict[driver] = [total_laps, completed_laps, percentage]

    laps_dict = dict(sorted(laps_dict.items(), key=lambda item: item[1][2], reverse=True))
    count = 1
    for d, l in laps_dict.items():
        print(f'{count}: {d} - {l[2]}% ({l[1]}-{l[0]})')
        count += 1


def winning_positions_per_circuit(circuit, start=1950, end=2024):
    """
    Return the winning positions from each year for a circuit
    :param circuit: circuit
    :param start: year of start
    :param end: year of end
    :return:
    """

    ergast = My_Ergast()
    r = ergast.get_race_results([i for i in range(start, end)])
    positions_dict = {}
    for race in r.content:
        race_circuit = race['circuitRef'].min()
        if circuit == race_circuit:
            win_data = race[race['position'] == 1]
            grid_pos = win_data['grid'].min()
            year = win_data['year'].min()
            d_name = win_data['fullName'].min()
            if grid_pos in positions_dict:
                positions_dict[grid_pos].append(f'{year}: {d_name}')
            else:
                positions_dict[grid_pos] = [f'{year}: {d_name}']
    positions_dict = dict(sorted(positions_dict.items()))
    for key, values in positions_dict.items():
        print(f'FROM P{key}:')
        for v in values:
            print(v)


def q3_appearances(year):
    ergast = My_Ergast()
    q = ergast.get_qualy_results([year])
    drivers_dict = {}
    for qualy in q.content:
        q_drivers = qualy['fullName'].values
        for d in q_drivers:
            qualy_data = qualy[qualy['fullName'] == d]
            position = qualy_data['position'].min()
            if position <= 10:
                if d in drivers_dict:
                    drivers_dict[d] += 1
                else:
                    drivers_dict[d] = 1
    drivers_dict = dict(sorted(drivers_dict.items(), key=lambda item: item[1], reverse=True))
    grouped_dict = defaultdict(list)
    for key, value in drivers_dict.items():
        grouped_dict[value].append(key)
    for v, d in grouped_dict.items():
        d = ', '.join(d)
        print(f'{v} - {d}')


def results_from_pole(driver, start=1950, end=2024, grid=1):
    ergast = My_Ergast()
    r = ergast.get_race_results([i for i in range(start, end)])
    finish_positions = Counter()
    winners = Counter()
    for race in r.content:
        pole = race[race['grid'] == grid]
        pole = pole[pole['fullName'] == driver]
        if len(pole) == 1:
            status = pole['status'].values[0]
            if re.search(r'(Finished|\+)', status):
                finish_pos = f'P{pole["position"].values[0]}'
            else:
                finish_pos = 'DNF'
            finish_positions[finish_pos] += 1
            winner = race[race['position'] == 1]
            winners[winner['fullName'].iloc[0]] += 1
            print(f'{pole["year"].values[0]} {pole["raceName"].values[0]}: From P{grid} to {finish_pos}')

    for pos, count in sorted(finish_positions.items(), key=lambda x: x[1], reverse=True):
        print(f'{pos}: {count} times')

    for pos, count in sorted(winners.items(), key=lambda x: x[1], reverse=True):
        print(f'{pos}: {count} times')

    def sankey_plot(counter, driver, title):
        drivers_expanded = [[f'{driver} ({count})'] * count for driver, count in counter.items()]
        drivers_flat = [driver for sublist in drivers_expanded for driver in sublist]
        df = pd.DataFrame(drivers_flat, columns=['Winner'])
        df['Driver'] = f'{driver} ({len(df)})'
        df['Count'] = df['Winner'].apply(lambda x: int(x.split('(')[1].split(')')[0]))
        df = df.sort_values(by='Count', ascending=True)
        df = df.drop('Count', axis=1)
        sankey.sankey(df['Driver'], df['Winner'], aspect=1000, fontsize=16)
        plt.title(title, font='Fira Sans', fontsize=18, color='white', ha='center')
        plt.tight_layout()
        plt.savefig(f'../PNGs/{title}.png', dpi=450)
        plt.show()

    sankey_plot(winners, driver, f'RACE WINNERS WITH {driver.upper()} ON POLE')
    sankey_plot(finish_positions, driver, f'{driver.upper()} RESULTS STARTING ON POLE')






def highest_qualy(team, start, end=2024):
    ergast = My_Ergast()
    q = ergast.get_qualy_results([i for i in range(start, end)])
    max_pos = 50
    race = None
    for qualy in q.content:
        team_data = qualy[qualy['constructorRef'] == team]
        if len(team_data) == 0:
            print(f'No data for {team} in {qualy["year"].min()}')
        else:
            q_pos = team_data['position'].min()
            if q_pos < max_pos:
                max_pos = q_pos
                race = f'{team_data["year"].min()} - {team_data["raceName"].min()}'
            elif q_pos == max_pos:
                race += f'{team_data["year"].min()} - {team_data["raceName"].min()} \n'

    print(max_pos, race)


def last_result_grid_pos(driver, grid_pos):
    ergast = My_Ergast()
    r = ergast.get_race_results([i for i in range(1950, 2023)])
    r.content.reverse()
    for race in r.content:
        d_data = race[race['fullName'] == driver]
        if len(d_data) == 1:
            if d_data['grid'].iloc[0] == grid_pos:
                print(f'{d_data["year"].iloc[0]} - {d_data["raceName"].iloc[0]}: From '
                      f'{d_data["grid"].iloc[0]} to {d_data["position"].iloc[0]}')
                break

def comebacks_in_circuit(circuit):
    ergast = My_Ergast()
    r = ergast.get_race_results([i for i in range(1950, 2023)])
    comeback_dict = {}
    for race in r.content:
        if race["circuitRef"].iloc[0] == circuit:
            race = race[~race['status'].str.contains('Did')].copy()
            race.loc[:, 'ogGrid'] = race['grid']
            race.loc[:, 'grid'] = race['grid'].replace(0, len(race))
            race.loc[:, 'posChange'] = race['grid'] - race['position']
            top_3 = sorted(set(race['posChange'].values), reverse=True)[:3]
            for v in top_3:
                comeback = race[race['posChange'] == v]
                for i in range(len(comeback)):
                    c_data = (f'{comeback["fullName"].iloc[i]}: From P{comeback["ogGrid"].iloc[i]}'
                              f' to P{comeback["position"].iloc[i]} ({comeback["year"].iloc[i]})//')
                    if v in comeback_dict:
                        comeback_dict[v] = comeback_dict[v] + c_data
                    else:
                        comeback_dict[v] = c_data

    comeback_dict = dict(sorted(comeback_dict.items(), key=lambda item: item[0], reverse=True))
    for key, value in comeback_dict.items():
        value = value.split('//')
        for v in value:
            if v != '':
                print(f'{key} places - {v}')




def driver_grid_positions(driver, start=1950, end=2024):
    ergast = My_Ergast()
    r = ergast.get_race_results([i for i in range(start, end)])
    grid_positions = Counter()
    for race in r.content:
        data = race[race['position'] == 1]
        data = data[data['fullName'] == driver]
        if len(data) == 1:
            grid_pos = data['grid'].iloc[0]
            grid_positions[grid_pos] += 1
            print(f'{grid_pos} - {race["year"].iloc[0]}: {race["raceName"].iloc[0]}')

    for pos, count in sorted(grid_positions.items(), key=lambda x: x[1], reverse=True):
        print(f'{pos}: {count} times')

