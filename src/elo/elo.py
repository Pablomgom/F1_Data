import math
import re
import time

import numpy as np
from fastf1.ergast import Ergast


class Driver:
    def __init__(self, name, total_races, initial_rating=1500):
        self.name = name
        self.previous_rating = initial_rating
        self.rating = initial_rating
        self.elo_changes = 0
        self.historical_elo = [0 for i in range(total_races)]
        self.num_races = 0
        self.races_factor = 0


def calculate_elo(r1, r2, s1, weight, k=35):
    e1 = 1 / (1 + 10 ** ((r2 - r1) / 250))
    return k * (s1 - e1) * weight


def get_pos(given_name, family_name, race_results):
    driver_data = race_results[(race_results['givenName'] == given_name) & (race_results['familyName'] == family_name)]
    if len(driver_data) > 0:
        return driver_data['position'].max()
    else:
        return 0


def get_avg_teamamte_elo(drivers, teammates):
    avg_elo = []
    for driver in drivers:
        avg_elo.append(driver.previous_rating)
    return np.mean(avg_elo)


def get_avg_teammate_pos(given_name, family_name, results, drivers, da_pos):
    teams = set(
        results[(results['givenName'] == given_name) & (results['familyName'] == family_name)]['constructorId'].values)
    avg_pos = []
    teammates_to_delete = []
    for team in teams:
        avg_team_pos = []
        teammates = set(results[(results['constructorId'] == team)
                                & (results['givenName'] != given_name)
                                & (results['familyName'] != family_name)
                                ]['driverId'].values)
        for teammate in teammates:
            teammate_pos = results[results['driverId'] == teammate]['position'].max()
            teammate_status = results[results['position'] == teammate_pos]['status'].values[0]
            if get_finish_status(status=teammate_status):
                avg_team_pos.append(teammate_pos)
            else:
                teammates_to_delete.append(teammate)
        if len(avg_team_pos) > 0:
            avg_team_pos.append(da_pos)
            avg_pos.append(np.mean(avg_team_pos))
    if len(avg_pos) == 0:
        return -1, -1
    for d_remove in teammates_to_delete:
        teammates.remove(d_remove)
    full_names = []
    for driver in teammates:
        teammate = results[results['driverId'] == driver]
        full_name = teammate['givenName'].values[0] + '//' + teammate['familyName'].values[0]
        full_names.append(full_name)
    drivers_to_check = []
    for full_name in full_names:
        for teammate in drivers:
            if full_name == teammate.name:
                drivers_to_check.append(teammate)
                break
    avg_elo = get_avg_teamamte_elo(drivers_to_check, teammates)
    return round(np.mean(avg_pos), 2), avg_elo


def get_finish_status(given_name='', family_name='', results=None, status=None):
    if status is None:
        driver_result = results[(results['givenName'] == given_name) & (results['familyName'] == family_name)]
        driver_result = driver_result.sort_values(by='position', ascending=False)
        status = driver_result['status'].values[0]
    '''
    if status not in ['Accident', 'Collision', 'Spun off', 'Injured', 'Injury',
                      'Fatal accident', 'Collision damaga', 'Damage', 'Physical']:
        if not re.search(r'(Finished|\+)', status):
            return False
    return True
    '''
    if not re.search(r'(Finished|\+)', status):
        return False
    return True


def race_influence_factor(num_races):
    a = 10
    b = 1
    return a / (num_races + b)


def min_max_scale_factors(factors, scale_min=1, scale_max=2):
    min_val = min(factors)
    max_val = max(factors)

    if min_val == max_val:
        return [scale_min for _ in factors]

    return [scale_min + (f - min_val) * (scale_max - scale_min) / (max_val - min_val) for f in factors]


def normalize_factors(factors):
    factor_sum = sum(factors)
    return [f / factor_sum for f in factors]



def update_ratings(drivers, race_results, race_index):
    pos_weight = [0.15 * (0.85 ** i) for i in range(len(race_results))]
    team_weight = [0.5 + (1 - 0.5) * (i / (5 - 1)) for i in range(5)]
    factors = [race_influence_factor(d.num_races) for d in drivers]
    scaled_factors = min_max_scale_factors(factors)
    normalized_factors = normalize_factors(scaled_factors)
    for i, d_a in enumerate(drivers):
        d_a.races_factor = normalized_factors[i]

    for d_a in drivers:
        d_a_parts_name = d_a.name.split('//')
        if get_finish_status(d_a_parts_name[0], d_a_parts_name[1], race_results):
            d_a_pos = get_pos(d_a_parts_name[0], d_a_parts_name[1], race_results)
            for d_b in drivers:
                if d_a.name != d_b.name:
                    d_b_parts_name = d_b.name.split('//')
                    if get_finish_status(d_b_parts_name[0], d_b_parts_name[1], race_results):
                        d_b_pos = get_pos(d_b_parts_name[0], d_b_parts_name[1], race_results)
                        if d_a_pos != 0 and d_b_pos != 0:
                            w_pos_a = pos_weight[d_a_pos - 1]
                            w_pos_b = pos_weight[d_b_pos - 1]
                            weight = abs(w_pos_a - w_pos_b)
                            if w_pos_a > w_pos_b:
                                s_a = 1
                            elif w_pos_b > w_pos_a:
                                s_a = 0
                            else:
                                s_a = 0.5

                            d_a.elo_changes += round(calculate_elo(d_a.previous_rating,
                                                                   d_b.previous_rating, s_a, weight, 5), 2)

            avg_team_pos = get_avg_teammate_pos(d_a_parts_name[0], d_a_parts_name[1], race_results, drivers, d_a_pos)
            if avg_team_pos[0] != -1:
                if d_a_pos < avg_team_pos[0]:
                    s_a = 1
                elif d_a_pos > avg_team_pos[0]:
                    s_a = 0
                else:
                    s_a = 0.5
                pos_dif = abs(d_a_pos - avg_team_pos[0])
                if pos_dif < len(team_weight) and pos_dif != 0:
                    weight = team_weight[int(math.ceil(pos_dif)) - 1]
                    num_d = len(race_results[race_results['position'] == d_a_pos])
                    weight = weight / num_d
                    if (pos_dif - int(pos_dif)) != 0:
                        if (pos_dif + 1) >= len(team_weight):
                            next_pos = 1.2
                        else:
                            next_pos = team_weight[math.ceil(pos_dif)]
                        weight += (next_pos - 1) * (pos_dif - int(pos_dif))
                elif pos_dif > len(team_weight):
                    weight = team_weight[len(team_weight) - 1]
                else:
                    weight = team_weight[0]
                    num_d = len(race_results[race_results['position'] == d_a_pos])
                    weight = weight/num_d
                    if num_d > 1:
                        s_a = 0

                d_a.elo_changes += round(calculate_elo(d_a.previous_rating,
                                                       avg_team_pos[1], s_a, weight, 35), 2)


    for driver in drivers:
        driver.rating += driver.elo_changes
        driver.historical_elo[race_index] = driver.rating
        driver.elo_changes = 0
        driver.previous_rating = driver.rating


def elo_execution(start, end):
    ergast = Ergast()
    races = []
    for year in range(start, end):
        year_data = ergast.get_race_results(season=year, limit=1000).content
        races += [race for race in year_data]
    driver_names = set([code for race in races for code in race['givenName'] + '//' + race['familyName']])
    drivers = [Driver(name, len(races)) for name in driver_names]
    driver_names = [d.name for d in drivers]
    race_index = 0
    for race in races:
        drivers_in_race = [code for code in race['givenName'] + '//' + race['familyName']]
        intersection = [value for value in driver_names if value in drivers_in_race]
        drivers_in_race = [d for d in drivers if d.name in intersection]
        for d in drivers_in_race:
            d.num_races += 1
        update_ratings(drivers_in_race, race, race_index)
        print(f'{race_index}/{len(races)}')
        drivers = sorted(drivers, key=lambda driver: max(driver.historical_elo), reverse=True)
        '''
        count = 0
        for driver in drivers:
            max_rating = round(driver.rating, 2)
            count += max_rating

        print(count)
        print(count / len(drivers))
        '''
        race_index += 1

    drivers = sorted(drivers, key=lambda driver: driver.rating, reverse=True)
    count = 0
    for driver in drivers:
        driver.rating = round(driver.rating, 2)
        count += driver.rating
        print(f'{driver.name} - {driver.rating}')
    print(count)
    print(count / len(drivers))
    print('----------------------------------------------------------------------')
    drivers = sorted(drivers, key=lambda driver: max(driver.historical_elo), reverse=True)
    for driver in drivers:
        max_rating = round(max(driver.historical_elo), 2)
        count += max_rating
        print(f'{driver.name} - {max_rating}')
