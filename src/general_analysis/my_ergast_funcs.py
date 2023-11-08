import re
from collections import defaultdict
from src.ergast_api.my_ergast import My_Ergast


def compare_my_ergast_teammates(given, family, start=2001, end=2024):
    """
    Compare a driver against his teammates
    :param given: Name
    :param family: Surname
    :param start: Year of start
    :param end: Year of end
    :return: None
    """

    def process_data(session, d_data, t_data, col, race_data):
        driver_data = session[(session['givenName'] == given) & (session['familyName'] == family)]
        if len(driver_data) == 1:
            team = driver_data['constructorName'].values[0]
            team_data = session[session['constructorName'] == team]
            team_data = team_data[(team_data['givenName'] != given) & (team_data['familyName'] != family)]
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
                driver_race_data = race_data[(race_data['givenName'] == given) & (race_data['familyName'] == family)]
                team_race_data = race_data[race_data['constructorName'] == team]
                team_race_data = team_race_data[
                    (team_race_data['givenName'] != given) & (team_race_data['familyName'] != family)]
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
        driver_data = race[(race['givenName'] == given) & (race['familyName'] == family)]
        if len(driver_data) == 1:
            team = driver_data['constructorName'].values[0]
            team_data = race[race['constructorName'] == team]
            team_data = team_data[(team_data['givenName'] != given) & (team_data['familyName'] != family)]
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


def results_from_pole(driver, start=1950, end=2024):

    ergast = My_Ergast()
    r = ergast.get_race_results([i for i in range(start, end)])
    for race in r.content:
        pole = race[race['grid'] == 1]
        pole = pole[pole['fullName'] == driver]
        if len(pole) == 1:
            status = pole['status'].values[0]
            if re.search(r'(Finished|\+)', status):
                finish_pos = f'P{pole["position"].values[0]}'
            else:
                finish_pos = 'DNF'

            print(f'{pole["year"].values[0]} {pole["raceName"].values[0]}: From P1 to {finish_pos}')


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