import math
import pickle
import re
import numpy as np
from fastf1.ergast import Ergast

from src.ergast_api.my_ergast import My_Ergast


class Driver:
    def __init__(self, name, initial_rating=1500):
        self.name = name
        self.previous_rating = initial_rating
        self.rating = initial_rating
        self.elo_changes = 0
        self.historical_elo = {}
        self.num_races = 0
        self.races_factor = 0


def calculate_elo(r1, r2, s1, weight, k=35):
    e1 = 1 / (1 + 10 ** ((r2 - r1) / 250))
    return k * (s1 - e1) * weight


def get_pos(given_name, family_name, race_results):
    driver_data = race_results[(race_results['givenName'] == given_name) & (race_results['familyName'] == family_name)]
    if len(driver_data) > 0:
        return driver_data['position'].min()
    else:
        return 0


def get_avg_teamamte_elo(drivers, teammates):
    avg_elo = []
    for driver in drivers:
        avg_elo.append(driver.previous_rating)
    return np.mean(avg_elo)


def get_avg_teammate_pos(given_name, family_name, results, drivers, da_pos):
    teams = set(
        results[(results['givenName'] == given_name) & (results['familyName'] == family_name)]['constructorName'].values)
    avg_pos = []
    teammates_to_delete = []
    for team in teams:
        avg_team_pos = []
        teammates = set(results[(results['constructorName'] == team)
                                & (results['givenName'] != given_name)
                                & (results['familyName'] != family_name)
                                ]['driverRef'].values)
        for teammate in teammates:
            teammate_pos = results[results['driverRef'] == teammate]['position'].max()
            teammate_status = results[results['position'] == teammate_pos]['status'].max()
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
        teammate = results[results['driverRef'] == driver]
        full_name = teammate['givenName'].max() + '//' + teammate['familyName'].max()
        full_names.append(full_name)
    drivers_to_check = []
    for full_name in full_names:
        for teammate in drivers:
            if full_name == teammate.name:
                drivers_to_check.append(teammate)
                break
    avg_elo = get_avg_teamamte_elo(drivers_to_check, teammates)
    return np.mean(avg_pos), avg_elo


def get_finish_status(given_name='', family_name='', results=None, status=None):
    if status is None:
        driver_result = results[(results['givenName'] == given_name) & (results['familyName'] == family_name)]
        driver_result = driver_result.sort_values(by='position', ascending=False)
        status = driver_result['status'].max()
    if not re.search(r'(Finished|\+)', status):
        return False
    return True


def is_accident(given_name='', family_name='', results=None, status=None):
    if status is None:
        driver_result = results[(results['givenName'] == given_name) & (results['familyName'] == family_name)]
        driver_result = driver_result.sort_values(by='position', ascending=False)
        status = driver_result['status'].max()
    if status in ['Accident', 'Collision', 'Spun off', 'Injured', 'Injury',
                  'Fatal accident', 'Collision damage', 'Damage', 'Physical']:
        return True
    return False


def race_influence_factor(num_races):
    a = 10
    b = 1
    return a / (num_races + b)


def min_max_scale_factors(factors, year, scale_min=1, scale_max=2.5):
    min_val = min(factors)
    max_val = max(factors)

    if min_val == max_val:
        return [scale_min for _ in factors]

    if year < 1960:
        scale_max = 5
    elif year < 1970:
        scale_max = 4.5
    elif year < 1980:
        scale_max = 4.5
    elif year < 1990:
        scale_max = 4.5
    elif year < 2000:
        scale_max = 4
    elif year < 2010:
        scale_max = 3.5

    return [scale_min + (f - min_val) * (scale_max - scale_min) / (max_val - min_val) for f in factors]


def normalize_factors(factors):
    total_drivers = len(factors)
    factor_sum = sum(factors)
    return [f * total_drivers / factor_sum for f in factors]


def get_valid_drivers(drivers, results, year):
    valid_drivers = []
    for d in drivers:
        name = d.name.split('//')
        if get_finish_status(name[0], name[1], results):
            valid_drivers.append(d)
        else:
            if is_accident(name[0], name[1], results) and year >= 2000:
                team = results[(results['givenName'] == name[0])
                               & (results['familyName'] == name[1])]['constructorName'].max()
                team_results = results[results['constructorName'] == team]
                teammate_data = team_results[(team_results['givenName'] != name[0])
                                             & (team_results['familyName'] != name[1])]
                if len(teammate_data) == 1:
                    teamamte_name = teammate_data['givenName'].max() + '//' + teammate_data['familyName'].max()
                    teammate_status = teammate_data['status'].max()
                    if get_finish_status(status=teammate_status) or not is_accident(status=teammate_status):
                        d.elo_changes -= 7.5
                        for d_t in drivers:
                            if d_t.name == teamamte_name:
                                d_t.elo_changes += 7.5
                                break
    return valid_drivers


def update_ratings(drivers, race_results, race_index, race_name):
    if '500' not in race_name:
        pos_weight = [0.2 * (0.85 ** i) for i in range(len(race_results))]
        team_weight = [0.5 + (1.5 - 0.5) * (i / (5 - 1)) for i in range(5)]
        current_year = int(race_name.split(' ')[0])
        drivers_available = get_valid_drivers(drivers, race_results, current_year)

        factors = [race_influence_factor(d.num_races) for d in drivers_available]
        scaled_factors = min_max_scale_factors(factors, current_year)
        normalized_factors = normalize_factors(scaled_factors)

        years = np.array(list(range(1950, 2025)))
        n_drivers = np.array(list(range(1, len(drivers) + 1)))
        values_k_pos = np.linspace(25, 7.5, len(years))
        values_k_team = np.linspace(82.5, 35, len(years))
        values_extra_weight_team = np.linspace(1.5, 0.5, len(drivers_available))

        k_pos = {year: value for year, value in zip(years, values_k_pos)}
        k_team = {year: value for year, value in zip(years, values_k_team)}
        k_extra_weight = {year: value for year, value in zip(n_drivers, values_extra_weight_team)}

        for i, d_a in enumerate(drivers_available):
            d_a.races_factor = normalized_factors[i]

        for d_a in drivers_available:
            d_a_parts_name = d_a.name.split('//')
            d_a_pos = get_pos(d_a_parts_name[0], d_a_parts_name[1], race_results)
            for d_b in drivers_available:
                if d_a.name != d_b.name:
                    d_b_parts_name = d_b.name.split('//')
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
                        d_a.elo_changes += calculate_elo(d_a.previous_rating,
                                                         d_b.previous_rating, s_a, weight,
                                                         k_pos[current_year]) * d_a.races_factor

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
                            next_pos = 1.8
                        else:
                            next_pos = team_weight[math.ceil(pos_dif)]
                        weight += (next_pos - 1) * (pos_dif - int(pos_dif))
                elif pos_dif > len(team_weight):
                    weight = team_weight[len(team_weight) - 1]
                else:
                    weight = team_weight[0]
                    num_d = len(race_results[race_results['position'] == d_a_pos])
                    weight = weight / num_d
                    if num_d > 1:
                        s_a = 0
                if round(avg_team_pos[0], 0) - 1 in list(k_extra_weight.keys()):
                    weight = weight * k_extra_weight[round(avg_team_pos[0], 0) - 1]
                else:
                    weight = weight * k_extra_weight[len(k_extra_weight)]
                    print(f'{d_a.name} - {round(avg_team_pos[0], 0)} - {max(list(k_extra_weight.keys()))}')
                d_a.elo_changes += calculate_elo(d_a.previous_rating,
                                                 avg_team_pos[1], s_a, weight,
                                                 k_team[current_year]) * d_a.races_factor

        gain_elo = []
        lose_elo = []

        for driver in drivers:
            elo_change = driver.elo_changes
            if elo_change < 0:
                lose_elo.append(elo_change)
            elif elo_change > 0:
                gain_elo.append(elo_change)

        diff_elo = sum(gain_elo) - abs(sum(lose_elo))
        reverse = True
        if diff_elo > 0:
            reverse = False
        drivers = sorted(drivers, key=lambda driver: driver.elo_changes, reverse=reverse)
        count = 0
        total_elo_change = 0
        for driver in drivers:
            if count == 0:
                driver.elo_changes += -diff_elo
            driver.rating += driver.elo_changes
            driver.historical_elo[race_name] = driver.rating
            driver.previous_rating = driver.rating
            count += 1
            total_elo_change += driver.elo_changes
        print(total_elo_change)


def elo_execution(start=None, end=None, restore=False, year=None, round=None, only_print=False, save=False):
    """
        Calculates the elo of the grid in the given years

        Parameters:
        start (int): Start year
        end (int): End year

    """

    ergast = Ergast()

    if only_print:
        with open("elo/elo_data.pkl", "rb") as file:
            drivers = pickle.load(file)

    else:
        races = []
        races_name = []
        if restore:
            data = My_Ergast().get_race_results([year], round)
            races = [data.content[0]]
            races_name.append(f'{data.content[0]["year"].loc[0]} - {data.content[0]["raceName"].loc[0]}')
            with open("elo/elo_data.pkl", "rb") as file:
                drivers = pickle.load(file)
            with open("elo/elo_data_bk.pkl", "wb") as file:
                pickle.dump(drivers, file)
        else:
            for year in range(start, end):
                year_data = ergast.get_race_results(season=year, limit=1000)
                races += [race for race in year_data.content]
                for i in range(len(year_data.description)):
                    race_names = year_data.description['raceName'].values[i]
                    races_name.append(f'{year} - {race_names}')
            driver_names = set([code for race in races for code in race['givenName'] + '//' + race['familyName']])
            drivers = [Driver(name) for name in driver_names]
        for d in drivers:
            for unique_race in races_name:
                d.historical_elo[unique_race] = 0
        driver_names = [d.name for d in drivers]
        race_index = 0
        for race in races:
            drivers_in_race = [code for code in race['givenName'] + '//' + race['familyName']]
            intersection = [value for value in driver_names if value in drivers_in_race]
            new_drivers = [d for d in drivers_in_race if d not in driver_names]
            new_drivers = [Driver(name) for name in new_drivers]
            for d in new_drivers:
                for r in list(drivers[0].historical_elo.keys()):
                    d.historical_elo[r] = 0
            drivers_in_race = [d for d in drivers if d.name in intersection] + new_drivers
            for d in drivers_in_race:
                d.elo_changes = 0
                d.num_races += 1
            update_ratings(drivers_in_race, race, race_index, races_name[race_index])
            print(f'{race_index}/{len(races)}')
            race_index += 1
            drivers += new_drivers
    total_elo = 0
    for driver in drivers:
        total_elo += driver.rating
        driver.ma_elo = {}
        for key, value in driver.historical_elo.items():
            year = key.split(" ")[0]
            if year not in driver.ma_elo:
                driver.ma_elo[year] = []
            if value != 0:
                driver.ma_elo[year].append(value)
        for year, values in driver.ma_elo.items():
            if len(values) == 0:
                driver.ma_elo[year] = 0
            else:
                driver.ma_elo[year] = sum(values) / len(values)
        years_sorted = sorted(driver.ma_elo.keys())
        driver.ma_elo_3ma = {}
        for i, year in enumerate(years_sorted):
            if i >= 3:
                three_years = [years_sorted[i - j] for j in range(4)]
                avg = sum([driver.ma_elo[y] for y in three_years]) / 4
                driver.ma_elo_3ma[year] = avg
    print(total_elo / len(drivers))

    print('----------------------------------------------------------------------')

    drivers = sorted(drivers, key=lambda driver: driver.rating, reverse=True)
    count = 1
    for driver in drivers[:25]:
        driver.rating = np.round(driver.rating, 2)
        print(f'{count} - {driver.name} - {driver.rating}')
        count += 1

    print('----------------------------------------------------------------------')
    top_drivers = sorted(drivers, key=lambda driver: (
        max(driver.historical_elo.values()),  # Maximize the Elo rating
        min(driver.historical_elo)  # Minimize the key lexicographically
    ), reverse=True)[:25]
    for count, driver in enumerate(top_drivers, start=1):
        max_key, max_rating = max(
            driver.historical_elo.items(),
            key=lambda item: (item[1], item[0])
        )
        max_rating = np.round(max_rating, 2)
        print(f'{count} - {driver.name} - {max_rating} (from race: {max_key})')
    try:
        print('----------------------------------------------------------------------')
        drivers = sorted(drivers, key=lambda driver: max(driver.ma_elo_3ma.values()), reverse=True)
        count = 1
        for driver in drivers[:25]:
            max_key = max(driver.ma_elo_3ma, key=driver.ma_elo_3ma.get)
            max_rating = np.round(driver.ma_elo_3ma[max_key], 2)
            print(f'{count} - {driver.name} - {max_rating} (from year: {max_key})')
            count += 1
    except:
        print('Not enough data for the MA')

    print('----------------------------------------------------------------------')
    season_races = []
    if restore or only_print:
        current_season = My_Ergast().get_race_results([int(year)], round)
        season_races = current_season.content[0]
    else:
        current_season = ergast.get_race_results(season=end - 1, limit=1000)
        season_races += [race for race in current_season.content]
        season_races = season_races[-1]
    current_drivers_names = (season_races['givenName'] + '//' + season_races['familyName']).values

    current_drivers = sorted(drivers, key=lambda driver: driver.rating, reverse=True)
    prev_drivers = sorted(drivers, key=lambda driver: driver.historical_elo[list(drivers[0].historical_elo)[-2]], reverse=True)
    prev_drivers_names = [d.name for d in prev_drivers]
    count = 1
    for driver in current_drivers:
        if driver.name in current_drivers_names:
            prev_rank = prev_drivers_names.index(driver.name) + 1
            driver.rating = np.round(driver.rating, 2)
            prev_rating = np.round(driver.historical_elo[list(drivers[0].historical_elo)[-2]], 2)
            if prev_rating == 0:
                diff = np.round(driver.elo_changes, 2)
                prev_rating = np.round(driver.rating - diff, 2)
            else:
                diff = np.round(driver.rating - prev_rating, 2)
            print(f'{count}: {driver.name} - {prev_rating} -> {driver.rating}({diff}) - Prev: {prev_rank}')
            count += 1

    if restore or save:
        with open("elo/elo_data.pkl", "wb") as file:
            pickle.dump(drivers, file)


def best_season(season=2007):
    with open("elo/elo_data.pkl", "rb") as file:
        drivers = pickle.load(file)

    rookie_session = {}

    for d in drivers:
        prev_value = 0
        first_race = True
        for r, e in d.historical_elo.items():
            current_year = int(r.split(' - ')[0])
            if current_year == season:
                diff_prev = prev_value - e
                if diff_prev != 0 and e != 0:
                    if (current_year, d.name) not in rookie_session:
                        if prev_value == 0:
                            rookie_session[(current_year, d.name)] = [1500, e]
                        else:
                            rookie_session[(current_year, d.name)] = [prev_value, e]
                    else:
                        rookie_session[(current_year, d.name)].append(e)
            if e != 0:
                prev_value = e

    filtered_rookie_session = sorted(rookie_session.items(), key=lambda x: x[1][-1] - x[1][0], reverse=True)
    index = 1
    for d, e in filtered_rookie_session:
        print(f'{index} - {d[1].replace("//", " ")}: {(e[-1] - e[0]):.2f} (from {e[0]:.2f} to {e[-1]:.2f})')
        index += 1
