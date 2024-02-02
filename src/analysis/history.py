import math
import re
import statistics
from collections import Counter

import fastf1
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt, cm
from pySankey import sankey
from tabulate import tabulate
from unidecode import unidecode

from src.ergast_api.my_ergast import My_Ergast
from src.plots.plots import get_handels_labels, get_font_properties
from src.plots.table import render_mpl_table
from src.utils.utils import get_dot
from src.variables.driver_colors import driver_colors_historical


def get_retirements():
    """
        Get all retirements in F1 history

   """

    races = []
    ergast = Ergast()

    for i in range(1950, 2023):
        races.append(ergast.get_race_results(season=i, limit=1000))
        races.append(ergast.get_sprint_results(season=i, limit=1000))
        print(i)

    init = pd.Series(dtype=object)

    for season in races:
        for race in season.content:
            init = pd.concat([init, race['status']], ignore_index=True)

    status = init.value_counts()
    status = status[~status.index.str.contains('+', regex=False)]
    status = status.drop('Finished')

    N = 10
    top_N = status.nlargest(N)
    top_N['Other'] = status.iloc[N:].sum()

    figsize = (8, 8)
    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    total_value = top_N.sum()

    colormap = cm.get_cmap('tab20', len(top_N))
    colors = [colormap(i) for i in range(len(top_N))]

    top_N.plot.pie(ax=ax, autopct='%1.1f%%', labels=['' for _ in top_N.index], legend=False,
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, colors=colors)  # Set line color to black

    ax.legend(title="Cause of Retirement", loc="center left", labels=top_N.index, bbox_to_anchor=(0.8, 0.1))

    plt.title(f'Retirements in F1 History (Total: {total_value})', fontsize=16, color='white')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'../PNGs/Retirements in F1 History', dpi=400)
    plt.show()

    items_per_table = 25
    start = 11
    other_values = status.iloc[N:]
    chunks = np.array_split(other_values, np.ceil(len(other_values) / items_per_table))

    for i in range(len(chunks)):
        other_values = pd.DataFrame(chunks[i]).reset_index().rename(columns={0: 'Total times',
                                                                             'index': 'Retirement cause'})

        end = start + len(other_values)
        other_values['Ranking'] = range(start, end)
        other_values = other_values[['Ranking', 'Retirement cause', 'Total times']]

        render_mpl_table(other_values, diff_column='No', col_Widths=[0.1, 0.3, 0.15],
                         title=f'TOP {start} - {end} retirement cause')

        start = end
        plt.tight_layout()
        plt.savefig(f'../PNGs/TOP {start} - {end} retirement cause', dpi=400)
        plt.show()


def get_circuitos():
    """
        Get all circuits in F1 history
   """

    ergast = Ergast()
    tracks = []
    countries = []

    for i in range(1950, 2024):
        season_circuits = ergast.get_race_results(season=i, limit=10000)
        tracks.extend(season_circuits.description['circuitId'])
        countries.extend(season_circuits.description['country'])
        print(i)
    tracks = pd.Series(tracks).value_counts()
    print(tracks)
    countries = pd.Series(countries).value_counts()
    print(countries)


def wins_and_poles_circuit(circuit, start=1950, end=2050):
    """
        Get all wins and poles in a circuit

        Parameters:
        circuit (str): Circuit to analyze
        start (int): Year of start
        end (int): Year of end

   """

    ergast = My_Ergast()
    winners = pd.Series(dtype=str)
    poles = pd.Series(dtype=str)
    years = []
    races = ergast.get_race_results([i for i in range(start, end)])
    for r in races.content:
        has_raced = r[r['circuitRef'] == circuit]
        if len(has_raced) > 0:
            data = r[r['position'] == 1]
            qualy_data = r[r['grid'] == 1]
            driver_win = str(data['fullName'].loc[0])
            driver_pole = str(qualy_data['fullName'].loc[0])
            winners = pd.concat([winners, pd.Series([driver_win])], ignore_index=True)
            poles = pd.concat([poles, pd.Series([driver_pole])], ignore_index=True)
            years.append(r['year'].loc[0])

    winners.index = years
    poles.index = years

    winners = (winners.groupby(winners).agg(["count", lambda x: list(x.index)])
               .rename(columns={"count": "Times", "<lambda_0>": "Years"}))

    poles = (poles.groupby(poles).agg(["count", lambda x: list(x.index)])
             .rename(columns={"count": "Times", "<lambda_0>": "Years"}))

    winners['First_Year'] = winners['Years'].apply(lambda x: x[0])
    winners = winners.sort_values(by=["Times", "First_Year"], ascending=[False, False])
    winners = winners.drop(columns=["First_Year"])
    poles['First_Year'] = poles['Years'].apply(lambda x: x[0])
    poles = poles.sort_values(by=["Times", "First_Year"], ascending=[False, False])
    poles = poles.drop(columns=["First_Year"])

    print('POLES')
    for index, row in poles.iterrows():
        print(f'{row["Times"]} - {index}: {str(row["Years"]).replace("[", "(").replace("]", ")")}')
    print('WINNERS')
    for index, row in winners.iterrows():
        print(f'{row["Times"]} - {index}: {str(row["Years"]).replace("[", "(").replace("]", ")")}')


def get_historical_race_days():
    """
        Get the data of all the races in F1 history

   """

    dict = {}
    ergast = Ergast()
    for i in range(1950, 2024):
        schedule = ergast.get_race_schedule(season=i, limit=1000)
        for j in range(len(schedule)):
            date = schedule.raceDate[j]
            event = schedule.raceName[j] + ' - ' + str(i)
            dict[event] = date

        print(i)

    # Function to get month and day
    def get_month_day(timestamp):
        return timestamp.month, timestamp.day

    sorted_items = sorted(dict.items(), key=lambda item: get_month_day(item[1]))

    n_days = {}
    with open('../resources/txt/Races_by_day.txt', 'w') as file:
        for key, value in sorted_items:
            month, day = get_month_day(value)
            race_date = f'{month:02d}-{day:02d}'
            file.write(f'{race_date}: {key}\n')

            if race_date in n_days:
                current = n_days[race_date]
                n_days[race_date] = current + 1
            else:
                n_days[race_date] = 1

    # Sort items by value in descending order
    sorted_days = sorted(n_days.items(), key=lambda item: item[1], reverse=True)

    # Print each key-value pair on a new line
    with open('../resources/txt/Grouped_races_by_day.txt', 'w') as file:
        for key, value in sorted_days:
            file.write(f'{key}: {value}\n')


def races_by_driver_dorsal(number):
    """
        Get the drivers who raced with a given number

        Parameters:
        number (int): Dorsal to be analuzed

   """

    ergast = Ergast()
    drivers = pd.Series(dtype=str)
    for i in range(1950, 2024):
        season = ergast.get_race_results(season=i, limit=1000)
        for race in season.content:
            filter_number = race[race['number'] == number]
            if len(filter_number) > 0:
                driver = filter_number['givenName'].values[0] + ' ' + filter_number['familyName'].values[0]
                driver_series = pd.Series([driver])
                drivers = drivers._append(driver_series, ignore_index=True)
        print(i)

    drivers = drivers.value_counts()
    print(drivers)


def lucky_drivers(start=None, end=None):
    """
        Get the luck of all drivers

        start (int): Year of start
        end (int): Year of end

   """

    if start is None:
        start = 1950
    if end is None:
        end = 2024
    ergast = My_Ergast()
    drivers_array = []
    all_races = []
    for i in range(start, end):
        races = ergast.get_race_results([i]).content
        for race in races:
            all_races.append(race)
            drivers_names = (race['givenName'] + ' ' + race['familyName']).values
            for name in drivers_names:
                drivers_array.append(name)
        print(f'GET DATA {i}')

    unique_drivers = set(drivers_array)
    luck = {}
    for name in unique_drivers:
        luck[name] = 0

    for driver in unique_drivers:
        for race in all_races:
            race_data = race[
                (race['givenName'] == driver.split(' ')[0]) & (race['familyName'] == driver.split(' ', 1)[1])]
            if len(race_data) > 0:
                teams = set(race_data['constructorName'])
                for team in teams:
                    teammate = race[race['constructorName'] == team]
                    teammate = list(teammate['givenName'] + ' ' + teammate['familyName'].values)
                    if len(teammate) > 1:
                        loops = teammate.count(driver)
                        teammate = [x for x in teammate if x != driver]
                        teammate = set(teammate)
                        for i in range(loops):
                            for team_name in teammate:
                                teammate_data = race[(race['givenName'] == team_name.split(' ')[0]) & (
                                        race['familyName'] == team_name.split(' ', 1)[1])]
                                teammate_data = teammate_data[teammate_data['constructorName'] == team]
                                status_d1 = race_data[race_data['constructorName'] == team]['status'].values[i]
                                if len(teammate_data) > 0:
                                    for j in range(len(teammate_data)):
                                        status_d2 = teammate_data['status'].values[j]
                                        if re.search(r'(Spun off|Accident|Withdrew|Collision|Damage|Finished|Did|\+)',
                                                     status_d1):
                                            if not re.search(
                                                    r'(Spun off|Accident|Withdrew|Collision|Damage|Finished|Did|\+)',
                                                    status_d2):
                                                luck[driver] += 1
                                        else:
                                            if re.search(
                                                    r'(Spun off|Accident|Withdrew|Collision|Damage|Finished|Did|\+)',
                                                    status_d2):
                                                luck[driver] -= 1
        print(driver)

    # Sort dictionary by its values in ascending order
    sorted_data = {k: v for k, v in sorted(luck.items(), key=lambda item: item[1], reverse=False)}

    values = 0
    for key, value in sorted_data.items():
        print(f"{key}: {value}/{drivers_array.count(key)} ({value / drivers_array.count(key) * 100:.3f}%)")
        values += value
    print(values)


def wins_per_year(start=2001, end=2024, top_10=True, historical_drivers=False, victories=True):
    ergast = Ergast()
    last_race = ergast.get_race_results(season=end - 1, round=17, limit=1000).content[0]
    current_drivers = []
    if not historical_drivers:
        current_drivers = last_race.base_class_view['givenName'].values + ' ' + last_race.base_class_view[
            'familyName'].values
    races_df = None
    for year in range(start, end):
        races = ergast.get_race_results(season=year, limit=1000).content
        for race in races:
            race['Year'] = year
            race['Driver'] = race['givenName'] + ' ' + race['familyName']
            races_df = pd.concat([races_df, race], axis=0, ignore_index=True)
            if historical_drivers:
                drivers_names = (race['givenName'] + ' ' + race['familyName']).values
                for d_n in drivers_names:
                    current_drivers.append(d_n)
        print(year)

    positions = [1, 2, 3]
    y_label = 'Total podiums'
    if victories:
        positions = [1]
        y_label = 'Total wins'
    races_df = races_df[races_df['position'].isin(positions)]
    races_df = races_df[races_df['Driver'].isin(current_drivers)]
    races_df = races_df.sort_values(by='Year')
    races_df = races_df.groupby(['Driver', 'Year']).size().reset_index(name='Wins')

    all_drivers = races_df['Driver'].unique()
    all_combinations = [(driver, year) for driver in all_drivers for year in range(start, end)]
    new_df = pd.DataFrame(all_combinations, columns=['Driver', 'Year'])
    races_df = new_df.merge(races_df, on=['Driver', 'Year'], how='left')
    races_df['Wins'] = races_df['Wins'].fillna(0).astype(int)
    races_df['Cumulative Wins'] = races_df.groupby('Driver')['Wins'].cumsum()
    races_df = races_df.sort_values(by='Year', ascending=True)
    if top_10:
        top_10_drivers = races_df.groupby('Driver')['Cumulative Wins'].max().sort_values(ascending=False).index.values[
                         :8]
        races_df = races_df[races_df['Driver'].isin(top_10_drivers)]
    colors = []
    for key, value in driver_colors_historical.items():
        fastf1.plotting.DRIVER_COLORS[key] = value
    for d in races_df['Driver'].unique():
        colors.append(fastf1.plotting.DRIVER_COLORS[unidecode(d.lower())])

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (driver, color) in enumerate(zip(races_df['Driver'].unique(), colors)):
        driver_data = races_df[races_df['Driver'] == driver]
        if historical_drivers:
            ax.plot(driver_data['Year'], driver_data['Cumulative Wins'], label=driver, color=color, linewidth=4.5)
        else:
            ax.plot(driver_data['Year'], driver_data['Cumulative Wins'], label=driver, color=color,
                    linewidth=3.5, marker='o', markersize=7)
    font_ticks = get_font_properties('Fira Sans', 15)
    font_legend = get_font_properties('Fira Sans', 12)
    handles, labels = get_handels_labels(ax)
    last_values = []
    for d in labels:
        last_values.append(max(races_df[races_df['Driver'] == d]['Cumulative Wins'].values))
    colors = [line.get_color() for line in ax.lines]
    info = list(zip(handles, labels, colors, last_values))
    info.sort(key=lambda item: item[3], reverse=True)
    handles, labels, colors, last_values = zip(*info)
    labels = [f"{label} ({last_value:.0f})" for label, last_value in zip(labels, last_values)]

    plt.legend(handles=handles, labels=labels, prop=font_legend, loc="upper left", fontsize='x-large')
    if historical_drivers:
        ax.set_xlim(left=min(races_df[races_df['Cumulative Wins'] >= 1]['Year']) - 2)
    ax.set_xlabel('Year', font='Fira Sans', fontsize=18)
    ax.set_ylabel(y_label, font='Fira Sans', fontsize=18)
    ax.grid(linestyle='--')
    plt.xticks(fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)
    plt.savefig(f'../PNGs/Cumulative {y_label}.png', dpi=450)
    plt.tight_layout()
    plt.show()


def laps_led(start, end):
    laps_led = pd.read_csv('../resources/csv/Laps_led.csv')
    laps_led = laps_led[(laps_led['Season']) >= start & (laps_led['Season'] <= end)]
    grouped_laps = laps_led.groupby('Driver')['Laps'].sum().reset_index().sort_values(by='Laps', ascending=False)
    total_laps = laps_led['Laps'].sum()
    grouped_laps['Rank'] = grouped_laps['Laps'].rank(method='min', ascending=False)

    for index, row in grouped_laps.iterrows():
        print(f'{int(row["Rank"])} - {row["Driver"]}: {row["Laps"]} ({(row["Laps"] / total_laps) * 100:.2f}%)')


def points_percentage_diff():
    data_dict = {}
    ergast = Ergast()
    for year in range(1950, 2024):
        wdc = ergast.get_driver_standings(year)
        wdc = wdc.content[0].sort_values(by='points', ascending=False)
        wdc = pd.DataFrame(wdc[['givenName', 'familyName', 'points']])
        wdc = wdc.loc[0:1]
        points_diff = round((wdc.loc[1, 'points'] / wdc.loc[0, 'points']) * 100, 2)
        champion_name = f'{wdc.loc[0, "givenName"]} {wdc.loc[0, "familyName"]}'
        sub_champion_name = f'{wdc.loc[1, "givenName"]} {wdc.loc[1, "familyName"]}'
        champion_points = wdc.loc[0, "points"]
        sub_champion_points = wdc.loc[1, "points"]
        data_dict[
            f'{year}: {champion_name} ({champion_points}) to {sub_champion_name} ({sub_champion_points})'] = points_diff
        print(year)

    data_dict = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=False))

    for k, v in data_dict.items():
        print(f'{k} - ({v})%')


def victories_per_driver_team(start=2014, end=2024):
    ergast = My_Ergast()
    races = ergast.get_race_results([i for i in range(start, end)])
    df = pd.DataFrame(columns=['Driver', 'Team'])
    for r in races.content:
        r = r[r['position'] == 1]
        driver = r['fullName'].iloc[0]
        team = r['constructorName'].iloc[0]
        df.loc[len(df)] = [driver, team]

    df['Team'] = df['Team'].replace('Alpine F1 Team', 'Alpine').replace('Team Lotus', 'Lotus')
    df['Team'] = df['Team'].str.split('-').str[0]
    df['Count Driver'] = df.groupby('Driver')['Driver'].transform('count')
    df['Count Teams'] = df.groupby('Team')['Team'].transform('count')
    df['Percentage wins'] = (df['Count Teams'] / len(df)) * 100
    df['Driver'] = [f'{driver} ({count})' for driver, count in zip(df['Driver'], df['Count Driver'])]
    df_print = df.groupby('Team')[['Count Teams', 'Percentage wins']].first()
    df_print = df_print.sort_values(by='Percentage wins', ascending=False)
    team_pos = 1
    for index, row in df_print.iterrows():
        print(f'{team_pos} - {index}: {row["Percentage wins"]:.2f}% ({row["Count Teams"]:.0f}/{len(df)})')
        team_pos += 1
    df['Team'] = [f'{driver} ({count})' for driver, count in zip(df['Team'], df['Count Teams'])]
    df = df.sort_values(by='Count Teams', ascending=True)
    df = df.drop(['Count Teams', 'Count Driver'], axis=1)
    sankey.sankey(df['Driver'], df['Team'], aspect=1000, fontsize=10.5)
    plt.title(f'WINS PER TEAM ({start}-{end - 1})', font='Fira Sans', fontsize=18, color='white')
    plt.tight_layout()
    plt.savefig(f'../PNGs/WINNERS-TEAM IN PERIOD.png', dpi=450)
    plt.show()


def difference_q1():
    ergast = My_Ergast()
    qualys = ergast.get_qualy_results([i for i in range(2014, 2024)]).content
    delta_diff = {}
    for q in qualys:
        q = q[~pd.isna(q['q1'])]
        year = q["year"].loc[0]
        min_time = min(q['q1'])
        max_time = max(q['q1'])
        delta = (max_time - min_time) / min_time
        if year not in delta_diff:
            delta_diff[year] = [delta]
        else:
            delta_diff[year].append(delta)
    for y, d in delta_diff.items():
        print(f'{y}: '
              f'MEDIAN: {round(statistics.median(d), 3)}'
              f' MEAN: {round(statistics.mean(d), 3)}')


def difference_second_team():
    ergast = My_Ergast()
    qualys = ergast.get_qualy_results([i for i in range(2014, 2024)]).content
    delta_diff = {}
    for q in qualys:
        col = 'q3'
        year = q["year"].loc[0]
        if q["year"].loc[0] == 2015 and q["round"].loc[0] == 16:
            col = 'q2'
        fastest_team = q['constructorName'].loc[0]
        fastest_lap = q[col].loc[0]
        q = q[q['constructorName'] != fastest_team]
        second_fast_lap = q[col].loc[0]
        delta = ((second_fast_lap - fastest_lap) / fastest_lap) * 100
        if year not in delta_diff:
            delta_diff[year] = [delta]
        else:
            delta_diff[year].append(delta)
    for y, d in delta_diff.items():
        print(f'{y}: '
              f'MEDIAN: {round(statistics.median(d), 3)}%'
              f' MEAN: {round(statistics.mean(d), 3)}%')


def difference_P2():
    ergast = My_Ergast()
    qualys = ergast.get_qualy_results([i for i in range(2014, 2024)]).content
    delta_diff = {}
    for q in qualys:
        col = 'q3'
        race_name = q["raceName"].loc[0]
        year = q["year"].loc[0]
        if q["year"].loc[0] == 2015 and q["round"].loc[0] == 16:
            col = 'q2'
        driver = q['fullName'].loc[0]
        fastest_lap = q[col].loc[0]
        driver_p2 = q[q['position'] == 2]['fullName'].loc[0]
        second_fast_lap = q[q['position'] == 2][col].loc[0]
        delta = (second_fast_lap - fastest_lap).total_seconds()
        full_race_name = f'{year} {race_name}'
        delta_diff[full_race_name] = f'{delta} from {driver} to {driver_p2}'

    delta_diff = dict(sorted(delta_diff.items(), key=lambda item: item[1], reverse=True))
    for r, d, in delta_diff.items():
        print(f'{r}: {d}')


def team_gap_to_next_or_fastest(team, start=2014, end=2024):
    def get_team_order(teams):
        order = []
        for element in teams:
            if element not in order:
                order.append(element)
        return order

    ergast = My_Ergast()
    qualys = ergast.get_qualy_results([i for i in range(start, end)]).content
    races = ergast.get_race_results([i for i in range(start, end)]).content
    sessions = ['q3', 'q2', 'q1']
    for q, r in zip(qualys, races):
        year = q['year'].loc[0]
        race_name = q['raceName'].loc[0].replace('Grand Prix', 'GP')
        for s in sessions:
            if year in [1989]:
                teams = q['constructorName'].values
                team_order = get_team_order(teams)
                team_data = q[q['constructorName'] == team]
                team_lap = min(min(team_data['q1']), min(team_data['q2'])).total_seconds()
                rivals = q[q['constructorName'] != team]
                rivals_lap = min(rivals['q2'].loc[0], rivals['q1'].loc[0]).total_seconds()
                rival_team = rivals['constructorName'].loc[0]
                diff = rivals_lap - team_lap
                print(f'{team} {diff:.3f}s {"faster" if diff > 0 else "slower"} than {rival_team} in {race_name}'
                      f' (P{team_order.index(team) + 1})')
                break
            else:
                team_data = q[(~pd.isna(q[s])) & (q['constructorName'] == team)]
                if len(team_data) > 0:
                    teams = q.sort_values(by=s, ascending=True)['constructorName'].values
                    team_order = get_team_order(teams)
                    team_order_race = get_team_order(r['constructorName'].values)
                    fastest_from_team = team_data[s].loc[0].total_seconds()
                    rivals = q[(~pd.isna(q[s])) & (q['constructorName'] != team)]
                    rivals = rivals.sort_values(by=s, ascending=True)
                    next_team_lap = rivals[s].loc[0].total_seconds()
                    next_team = rivals['constructorName'].loc[0]
                    diff = next_team_lap - fastest_from_team
                    dot = get_dot(diff)
                    print(
                        f'{dot}{abs(diff):.3f}s {"faster" if diff > 0 else "slower"} than {next_team} in the {race_name}')
                    break


def times_lapped_per_team(start=2014, end=2024):
    races = My_Ergast().get_race_results([i for i in range(start, end)])
    dict_per_year = {}
    drivers_dict = {}
    for r in races.content:
        max_laps = r['laps'].loc[0]
        for index, row in r.iterrows():
            status = row['status']
            year = row['year']
            if '+' in status:
                times_lapped = int(status.replace('+', '').split(' ')[0])
                driver_laps = row['laps']
                classified = True if driver_laps >= math.ceil(max_laps * 0.9) else False
                if classified:
                    team = row['constructorName']
                    driver = row['fullName']
                    if (year, team) not in dict_per_year:
                        dict_per_year[(year, team)] = times_lapped
                    else:
                        dict_per_year[(year, team)] += times_lapped

                    if driver not in drivers_dict:
                        drivers_dict[driver] = times_lapped
                    else:
                        drivers_dict[driver] += times_lapped

    print('---PER YEAR---')
    dict_per_year = dict(sorted(dict_per_year.items(), key=lambda item: (item[0][0], item[1])))
    for t, l in dict_per_year.items():
        print(f'{t[0]} - {t[1]}: {l}')

    print('---ALL TIME---')
    dict_per_year = dict(sorted(dict_per_year.items(), key=lambda item: item[1]))
    for t, l in dict_per_year.items():
        print(f'{t[0]} - {t[1]}: {l}')

    print('---DRIVERS---')
    drivers_dict = dict(sorted(drivers_dict.items(), key=lambda item: item[1]))
    for t, l in drivers_dict.items():
        print(f'{t}: {l}')


def avg_position_season(season):
    races = My_Ergast().get_race_results([season])
    positions_dict = {}
    dnfs_dict = {}
    for r in races.content:
        for index, row in r.iterrows():
            status = row['status']
            name = row['fullName']
            if status == 'Finished' or '+' in status:
                position = row['position']
                if name not in positions_dict:
                    positions_dict[name] = [position]
                else:
                    positions_dict[name].append(position)
            else:
                if name not in dnfs_dict:
                    dnfs_dict[name] = 1
                else:
                    dnfs_dict[name] += 1

    df = pd.DataFrame.from_dict(positions_dict, orient='index').transpose()
    df_avg = df.mean().reset_index()
    df_avg.columns = ['Driver', 'AvgPosition']
    df_avg['Rank'] = df_avg['AvgPosition'].rank(method='min')
    df_avg = df_avg.sort_values(by='Rank')
    for index, row in df_avg.iterrows():
        driver = row['Driver']
        total_races = len(positions_dict.get(driver, 0)) + dnfs_dict.get(driver, 0)
        print(
            f"{int(row['Rank'])} - {driver}: {row['AvgPosition']:.2f} - "
            f"DNFs ({dnfs_dict.get(driver, 0)}/{total_races}"
            f" {(dnfs_dict.get(driver, 0) / total_races) * 100:.2f}%)")


def dfns_per_year(start=1950, end=2050):
    races = My_Ergast().get_race_results([i for i in range(start, end)])
    dnfs_dict = {}
    for r in races.content:
        for index, row in r.iterrows():
            status = row['status']
            year = row['year']
            if (status == 'Finished' or '+' in status) or status.lower() == 'withdraw' or status.lower() == 'illness':
                if year not in dnfs_dict:
                    dnfs_dict[year] = [0]
                else:
                    dnfs_dict[year].append(0)
            else:
                if year not in dnfs_dict:
                    dnfs_dict[year] = 1
                else:
                    dnfs_dict[year].append(1)

    dnfs_dict = dict(sorted(dnfs_dict.items(), key=lambda item: np.sum(item[1]) / len(item[1])))
    for y, d in dnfs_dict.items():
        print(f'{y}: {(np.sum(d) / len(d)) * 100:.2f}% - Total ({np.sum(d)})')


def pole_position_evolution(circuit, start=1950, end=2050):

    qualys = My_Ergast().get_qualy_results([i for i in range(start, end)])
    prev_value = None
    log_entries = []
    for q in qualys.content:
        if q["circuitRef"].loc[0] == circuit:
            year = q["year"].loc[0]
            q['bestTime'] = q.apply(lambda row: min(row['q1'], row['q2'], row['q3']), axis=1)
            q = q.sort_values(by='bestTime', ascending=True).reset_index(drop=True)
            pole_time = q['bestTime'].loc[0]
            driver = q['fullName'].loc[0]
            hours, minutes, full_seconds = str(pole_time).split(':')
            if '.' in full_seconds:
                seconds, milliseconds = full_seconds.split('.')
            else:
                seconds = full_seconds
                milliseconds = '000'
            milliseconds_formatted = f"{float('0.' + milliseconds):.3f}".split('.')[1]
            formatted_pole_time = f"{minutes}:{seconds}.{milliseconds_formatted}"

            if prev_value is None:
                entry = f'{"🟠"}{year}: {formatted_pole_time} - {driver}'
            else:
                diff = (pole_time - prev_value).total_seconds()
                entry = f'{"🟢" if diff < 0 else "🔴"}{year}: {formatted_pole_time} - {driver} ({"+" if diff > 0 else ""}{diff:.3f}s)'
            prev_value = pole_time
            log_entries.append(entry)

    final_log = '\n'.join(log_entries)
    print(final_log)
