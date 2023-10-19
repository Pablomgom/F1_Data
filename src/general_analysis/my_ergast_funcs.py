import re

from src.ergast_api.my_ergast import My_Ergast


def compare_my_ergast_teammates(given, family, start=2001, end=2024):
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
                team_race_data = team_race_data[(team_race_data['givenName'] != given) & (team_race_data['familyName'] != family)]
                d_grid = driver_race_data['grid'].values[0]
                t_grid = team_race_data['grid'].values[0]
                if d_grid == 1:
                    d_data[1] += 1
                elif t_grid == 1:
                    t_data[1] += 1
                    print(f'{t_position} - {team_data["year"].min()} - {team_data["raceName"].min()}')

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
                #VICTORIES
                if d_position == 1:
                    d_data[3] += 1
                if t_position == 1:
                    t_data[3] += 1
                #PODIUMS
                if d_position in [1, 2, 3]:
                    d_data[4] += 1
                if t_position in [1, 2, 3]:
                    t_data[4] += 1
                #POINT FINISHES
                if d_points > 0:
                    d_data[5] += 1
                if t_points > 0:
                    t_data[5] += 1
                #TOTAL POINTS
                t_data[7] += t_points
                if re.search(r'(Finished|\+)', d_status) and re.search(r'(Finished|\+)', t_status):
                    if d_position < t_position:
                        d_data[2] += 1
                    else:
                        t_data[2] += 1

                else:
                    if not re.search(r'(Disqualified|Finished|\+)', d_status):
                        # print(f'{d_status} - {driver_data["year"].min()} - {driver_data["raceName"].min()}')
                        d_data[6] += 1
                    if not re.search(r'(Disqualified|Finished|\+)', t_status):
                        t_data[6] += 1
                        # print(f'{t_status} - {driver_data["year"].min()} - {driver_data["raceName"].min()}')

    print(d_data, t_data)




def get_driver_laps(year):

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
        percentage = round((completed_laps/total_laps) * 100, 2)
        laps_dict[driver] = [total_laps, completed_laps, percentage]

    laps_dict = dict(sorted(laps_dict.items(), key=lambda item: item[1][2], reverse=True))
    count = 1
    for d, l in laps_dict.items():
        print(f'{count}: {d} - {l[2]}%')
        count += 1
    a = 1