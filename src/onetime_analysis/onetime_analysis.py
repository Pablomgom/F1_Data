import math

import fastf1
import numpy as np
import pandas as pd
import re
import statistics

from adjustText import adjust_text
from fastf1 import plotting
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt, cm
from collections import Counter
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from statsmodels.tsa.arima.model import ARIMA

from src.plots.plots import rounded_top_rect, annotate_bars
from src.general_analysis.table import render_mpl_table

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from src.plots.plots import round_bars
from src.variables.variables import team_colors_2023, driver_colors_2023, point_system_2010, point_system_2009


def get_DNFs_team(team, start, end):
    ergast = Ergast()
    mechanical = 0
    accident = 0
    finished = 0
    for year in range(start, end):
        race_index = 0
        team_data = ergast.get_race_results(season=year, constructor=team, limit=1000)
        for race in team_data.content:
            finish_status = race['status'].values
            for status in finish_status:
                if re.search(r'(Spun off|Accident|Collision|Puncture)', status):
                    accident += 1
                elif re.search(r'(Finished|\+)', status):
                    finished += 1
                else:
                    mechanical += 1
                print(f'{year} - {team_data.description["circuitId"][race_index]} - {status}')
            race_index += 1
    print(f"""
        MECHANICAL: {mechanical}
        ACCIDENT: {accident}
        FINISHED: {finished}
    """)


def simulate_season_different_psystem(year, drivers):

    ergast = Ergast()
    for driver in drivers:
        points = 0
        races = ergast.get_race_results(season=year, driver=driver, limit=1000).content
        for race in races:
            pos = race['position'].values[0]
            if pos in list(point_system_2010.keys()):
                points += point_system_2010[pos]
            if 'fastestLapRank' in race.columns:
                if race['fastestLapRank'].values[0] == 1 and pos <= 10:
                    points += 1
        print(f'{driver} - {points}')

def simulate_qualy_championship(year):

    qualy_data = Ergast().get_qualifying_results(season=year, limit=1000)
    drivers = set([code for df in qualy_data.content for code in df['driverCode'].values])
    driver_points = {}
    for driver in drivers:
        driver_points[driver] = 0
    for qualy in qualy_data.content:
        for driver in drivers:
            driver_data = qualy[qualy['driverCode'] == driver]
            if len(driver_data) > 0:
                pos = qualy[qualy['driverCode'] == driver]['position'].values[0]
                if pos in list(point_system_2009.keys()):
                    driver_points[driver] += point_system_2009[pos]

    driver_points = dict(sorted(driver_points.items(), key=lambda item: item[1], reverse=True))
    total_p = 0
    for d, p in driver_points.items():
        print(f'{d} - {p}')
        total_p += p
    print(total_p)


def full_compare_drivers_season(year, d1, d2, team, mode=None, split=None, d1_team=None, d2_team=None):

    ergast = Ergast()
    if mode == 'team':
        race_results = ergast.get_race_results(season=year, constructor=team, limit=1000).content
        race_part_1 = [race_results[i] for i in range(split)]
        race_part_2 = [race_results[i] for i in range(split, len(race_results))]
        qualy_results = ergast.get_qualifying_results(season=year, constructor=team, limit=1000).content
        qualy_part_1 = [qualy_results[i] for i in range(split)]
        qualy_part_2 = [qualy_results[i] for i in range(split, len(qualy_results))]
        constructor_data_p1 = ergast.get_constructor_standings(season=year, constructor=team, round=split, limit=1000)
        constructor_data_p2 = ergast.get_constructor_standings(season=year, constructor=team, round=len(race_results), limit=1000)

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
        d1_race_results = ergast.get_race_results(season=year, driver=d1, limit=1000).content
        d1_code = d1_race_results[0]['driverCode'].values[0]
        d1_mean_race_pos_no_dnf = np.mean([i['position'].values[0] for i in d1_race_results
                                        if re.search(r'(Finished|\+)', i['status'].max())])

        d1_dnf_count = len([i['position'].values[0] for i in d1_race_results
                                        if not re.search(r'(Finished|\+)', i['status'].max())])

        d1_victories = len([i['position'].values[0] for i in d1_race_results if i['position'].values[0] == 1])
        d1_podiums = len([i['position'].values[0] for i in d1_race_results if i['position'].values[0] <= 3])

        d2_race_results = ergast.get_race_results(season=year, driver=d2, limit=1000).content
        d2_code = d2_race_results[0]['driverCode'].values[0]
        d2_mean_race_pos_no_dnf = np.mean([i['position'].values[0] for i in d2_race_results
                                        if re.search(r'(Finished|\+)', i['status'].max())])

        d2_dnf_count = len([i['position'].values[0] for i in d2_race_results
                        if not re.search(r'(Finished|\+)', i['status'].max())])

        d2_victories = len([i['position'].values[0] for i in d2_race_results if i['position'].values[0] == 1])
        d2_podiums = len([i['position'].values[0] for i in d2_race_results if i['position'].values[0] <= 3])

        d1_points = ergast.get_driver_standings(season=year, driver=d1, limit=1000).content[0]['points'].values[0]
        d2_points = ergast.get_driver_standings(season=year, driver=d2, limit=1000).content[0]['points'].values[0]

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
                    d1_lap_pos = d1_pos.values[lap]
                    d2_lap_pos = d2_pos.values[lap]
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
            DNFS: {d1} - {d1_dnf_count} --- {d2} - {d2_dnf_count}
            POINTS: {d1} - {d1_points} {d1_percentage}% --- {d2} - {d2_points} {d2_percentage}%
            LAPS IN FRONT: {d1} - {d1_laps_ahead} {d1_percentage_ahead}% --- {d2} - {d2_laps_ahead} {d2_percentage_ahead}%
        """)

def compare_qualy_results(team, threshold, end=None, exclude=None):
    if end is None:
        end = 1950
    ergast = Ergast()
    for year in range(2023, end, -1):
        qualys = ergast.get_qualifying_results(season=year, limit=1000)
        qualys_data = qualys.content
        qualys_data.reverse()
        circuits = list(qualys.description['circuitId'])
        circuits.reverse()
        for i in range(len(qualys_data)):
            exclude_qualy = False
            data = qualys_data[i]
            team_data = data[data['constructorId'] == team]
            check = data[data['constructorId'] == team]['position'].mean()
            print(f'{check} in {year} in {circuits[i]}')
            if exclude is not None:
                for circuit, year_dict in exclude.items():
                    if circuit == circuits[i] and year_dict == year:
                        exclude_qualy = True
            if exclude_qualy:
                continue
            if check >= threshold:
                print(f"""
                {year} in {circuits[i]}
                {list(team_data['driverId'])}
                {list(team_data['position'])}
                {check} points                
                """)
                exit(0)


def compare_amount_points(team, threshold, end=None, exclude=None):
    if end is None:
        end = 1950
    ergast = Ergast()
    for year in range(2023, end, -1):
        races = ergast.get_race_results(season=year, limit=1000)
        races_data = races.content
        races_data.reverse()
        circuits = list(races.description['circuitId'])
        circuits.reverse()
        for i in range(len(races_data)):
            data = races_data[i]
            team_data = data[data['constructorId'] == team]
            check = data[data['constructorId'] == team]['points'].sum()
            print(f'{check} in {year} in {circuits[i]}')
            if exclude is not None and check == exclude:
                break
            if check <= threshold:
                print(f"""
                {year} in {circuits[i]}
                {list(team_data['driverId'])}
                {list(team_data['position'])}
                {check} points                
                """)
                exit(0)


def race_qualy_avg_metrics(year, session='Q', predict=False, mode=None):
    ergast = Ergast()
    data = ergast.get_race_results(season=year, limit=1000)
    teams = set(data.content[0]['constructorName'])
    circuits = np.array(data.description['circuitId'])
    circuits = [i.replace('_', ' ').title() for i in circuits]
    team_points = pd.DataFrame(columns=['Team', 'Points', 'Circuit'])
    yticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    if session == 'Q':
        data = ergast.get_qualifying_results(season=year, limit=1000)
        yticks = [1, 3, 6, 9, 12, 15, 18, 20]
        title = 'Average qualy position in the last 4 races'
    else:
        if predict:
            title = 'Average points prediction in the last 4 races'
        else:
            title = 'Average points in the last 4 races'

    def append_duplicate_number(arr):

        # Dictionary to keep track of counts of appearances
        counts = {}

        # List to store processed items
        result = []

        for item in arr:
            # Check if item is duplicate
            if arr.count(item) > 1:
                # Increment the count for this item, or set to 1 if not seen before
                counts[item] = counts.get(item, 0) + 1
                # Append the count to the item and add to result
                result.append(f"{item} {counts[item]}")
            else:
                result.append(item)

        # Convert the result list back to ndarray
        result_arr = np.array(result)

        return result_arr

    circuits = append_duplicate_number(circuits)
    for i in range(len(data.content)):
        for team in teams:
            if circuits[i] == 'Red Bull Ring':
                a = 1
            if session == 'Q':
                points = data.content[i][data.content[i]['constructorName'] == team]['position'].mean()
            else:
                points = data.content[i][data.content[i]['constructorName'] == team]['points'].sum()
            row = [team, points, circuits[i]]
            team_points = team_points._append(pd.Series(row, index=team_points.columns), ignore_index=True)

    team_categories = pd.Categorical(team_points['Team'], categories=team_points['Team'].unique(), ordered=True)
    race_categories = pd.Categorical(team_points['Circuit'], categories=team_points['Circuit'].unique(), ordered=True)
    ct = pd.crosstab(team_categories, race_categories, values=team_points['Points'], aggfunc='sum')
    if mode is not None:
        ct = ct.cumsum(axis=1)
    else:
        ct = ct.rolling(window=4, min_periods=1, axis=1).mean()
    ordered_colors = [team_colors_2023[team] for team in ct.index]
    transposed = ct.transpose()
    if predict:
        forecasted_data = []
        for team in transposed.columns:
            model = ARIMA(transposed[team], order=(5, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=6)
            forecasted_data.append(forecast)

        forecasted_df = pd.DataFrame(forecasted_data).transpose()
        forecasted_df.columns = transposed.columns

        circuits = ergast.get_race_schedule(season=year, limit=1000)['circuitId'].values
        start_index = len(transposed)
        end_index = start_index + len(forecasted_df)
        new_indices = [circuits[i].title().replace('_', ' ') for i in range(start_index, end_index)]
        forecasted_df.index = new_indices

        transposed = pd.concat([transposed, forecasted_df])
        transposed = transposed.where(transposed >= 0, 0)

    ax = transposed.plot(figsize=(12, 12), marker='o', color=ordered_colors)

    if predict:
        start_x = len(transposed) - len(forecasted_df)  # Start of forecast
        end_x = len(transposed) - 1  # End of forecast (which is essentially the length of your combined data)
        ax.axvspan(start_x, end_x, facecolor='green', alpha=0.2)
        ax.annotate('Predictions', xy=((start_x + end_x) / 2, ax.get_ylim()[1] - 1),
                    xycoords='data', ha='center', fontsize=16, color='black', alpha=0.7,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))

    font = FontProperties(family='Fira Sans', size=12)
    plt.title(title, font='Fira Sans', fontsize=28)
    plt.xlabel("Races", font='Fira Sans', fontsize=18)
    plt.ylabel("Total Points", font='Fira Sans', fontsize=18)
    predict_patch = mpatches.Patch(color='green', alpha=0.5, label='Predictions')
    handles, labels = ax.get_legend_handles_labels()
    if predict:
        handles.append(predict_patch)
    plt.legend(handles=handles, prop=font, loc="upper left", bbox_to_anchor=(1.0, 0.6))
    plt.xticks(ticks=range(len(transposed)), labels=transposed.index,
               rotation=90, fontsize=12, fontname='Fira Sans')
    if mode is not None:
        plt.yticks(fontsize=12, fontname='Fira Sans')
    else:
        plt.yticks(yticks, fontsize=12, fontname='Fira Sans')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if session == 'Q':
        ax.invert_yaxis()
    plt.tight_layout()  # Adjusts the plot layout for better visibility
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.savefig(f'../PNGs/AVERAGE POINTS.png', dpi=400)
    plt.show()
    if predict:
        pd.options.display.max_colwidth = 50
        pd.options.display.width = 1000
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(transposed)


def plot_upgrades(scope=None):
    upgrades = pd.read_csv('../resources/upgrades.csv', sep='|')
    if scope is not None:
        upgrades = upgrades[upgrades['Reason'] == scope]
        upgrades = upgrades[upgrades['Race'] != 'Bahrain']
    team_categories = pd.Categorical(upgrades['Team'], categories=upgrades['Team'].unique(), ordered=True)
    race_categories = pd.Categorical(upgrades['Race'], categories=upgrades['Race'].unique(), ordered=True)
    ct = pd.crosstab(team_categories, race_categories)
    cumulative_sum = ct.cumsum(axis=1)
    ordered_colors = [team_colors_2023[team] for team in cumulative_sum.index]
    transposed = cumulative_sum.transpose()
    ax = transposed.plot(figsize=(10, 12), marker='o', color=ordered_colors)

    if scope is None:
        scope = ''
    else:
        scope += ' '
    plt.title(f"Cumulative {scope}Upgrades for Each Team", font='Fira Sans', fontsize=28)
    plt.xlabel("Races", font='Fira Sans', fontsize=18)
    plt.ylabel("Number of Upgrades", font='Fira Sans', fontsize=18)
    races = cumulative_sum.columns
    plt.xticks(ticks=range(len(races)), labels=races, rotation=90)

    # Initialize the previous y-value
    prev_y = None
    offset = 1
    # Annotate the last value of each line
    for team, color in zip(transposed.columns, ordered_colors):
        y_value = transposed[team].iloc[-1]
        if prev_y is not None and abs(prev_y - y_value) < offset:
            y_value += offset
        ax.annotate(f"{y_value:.0f}",
                    xy=(len(races) - 1, y_value),
                    xytext=(10, 0),  # 5 points horizontal offset
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    font='Fira Sans',
                    fontsize=13,
                    color=color)
        prev_y = y_value

    font = FontProperties(family='Fira Sans', size=12)
    plt.legend(prop=font, loc="upper left")
    plt.xticks(ticks=range(len(transposed)), labels=transposed.index,
               rotation=90, fontsize=12, fontname='Fira Sans')
    plt.yticks(fontsize=12, fontname='Fira Sans')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()  # Adjusts the plot layout for better visibility
    plt.savefig(f'../PNGs/{scope} UPGRADES.png', dpi=400)
    plt.show()
    pd.set_option('display.max_columns', None)
    print(transposed)


def cluster_circuits(year, rounds, prev_year=None, circuit=None, clusters=None):
    session_type = ['FP1', 'FP2', 'FP3', 'Q', 'S', 'SS', 'R']
    circuits = []
    data = []
    if prev_year is None:
        add_round = 0
    else:
        add_round = 1
    for i in range(0, rounds + add_round):
        prev_session = None
        fast_laps = []
        for type in session_type:
            try:
                if i == rounds:
                    session = fastf1.get_session(prev_year, circuit, type)
                    print(f'{i}: {type}')
                else:
                    session = fastf1.get_session(year, i + 1, type)
                session.load()
                prev_session = session
                fast_laps.append(session.laps.pick_fastest())
            except:
                print(f'{type} not in this event')

        session = prev_session
        fastest_lap = pd.Timedelta(days=1)
        telemetry = None
        for lap in fast_laps:
            if lap['LapTime'] < fastest_lap:
                try:
                    telemetry = lap.telemetry
                    fastest_lap = lap['LapTime']
                except:
                    print('Telemetry error')

        corners = len(session.get_circuit_info().corners)
        length = max(telemetry['Distance'])
        corners_per_meter = length / corners
        max_speed = max(telemetry['Speed'])
        sorted_speed = sorted(telemetry['Speed'])

        def get_percentile(data, percentile):
            size = len(data)
            return data[int((size - 1) * percentile)]

        def get_quartiles(data):
            data = sorted(data)

            # Calculate the median
            size = len(data)
            if size % 2 == 0:
                median = (data[size // 2 - 1] + data[size // 2]) / 2
            else:
                median = data[size // 2]

            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            lower_half = data[:size // 2] if size % 2 == 0 else data[:size // 2]
            upper_half = data[size // 2:] if size % 2 == 0 else data[size // 2 + 1:]

            q1 = get_percentile(lower_half, 0.5)
            q3 = get_percentile(upper_half, 0.5)

            avg = sum(data) / size

            return q1, median, q3, avg

        q1, median, q3, avg = get_quartiles(sorted_speed)

        full_gas = telemetry[telemetry['Throttle'].isin([100, 99])]
        full_gas = len(full_gas) / len(telemetry)
        lifting = telemetry[(telemetry['Throttle'] >= 1) & (telemetry['Throttle'] <= 99)]
        lifting = len(lifting) / len(telemetry)
        no_gas = telemetry[telemetry['Throttle'] == 0]
        no_gas = len(no_gas) / len(telemetry)
        brakes = telemetry[telemetry['Brake'] == 100]
        brakes = len(brakes) / len(telemetry)
        no_brakes = telemetry[telemetry['Brake'] == 0]
        no_brakes = len(no_brakes) / len(telemetry)
        data.append([corners_per_meter, max_speed, q1, median, q3, avg, full_gas, lifting])

        circuits.append(session.event.Location)

    data = StandardScaler().fit_transform(data)

    # Use PCA to reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    if clusters is None:
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        exit(0)

    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(data)

    # Calculate mean of the original features for each cluster
    cluster_means = {}
    for cluster_id in np.unique(y_kmeans):
        cluster_means[cluster_id] = data[y_kmeans == cluster_id].mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 8))

    texts = []
    colors = ['red', 'blue', 'green']  # Assuming 3 clusters; extend this list if there are more clusters

    for i, name in enumerate(circuits):
        ax.scatter(principal_components[i, 0], principal_components[i, 1], color=colors[y_kmeans[i]], s=100)
        texts.append(ax.text(principal_components[i, 0], principal_components[i, 1], name,
                             font='Fira Sans', fontsize=13))

    # Plotting centers and storing the text objects
    for i, center in enumerate(pca.transform(kmeans.cluster_centers_)):
        '''
        match i:
            case 0:
                type = 'Medium speed tracks'
            case 1:
                type = 'Low speed tracks'
            case 2:
                type = 'High speed tracks'
            case _:
                type = 'WTF IS A KILOMETER'
        '''

        texts.append(ax.text(center[0], center[1], type, font='Fira Sans',
                             fontsize=16, ha='right'))
        ax.scatter(center[0], center[1], s=300, c='#FF8C00')

    # Automatically adjust the positions to minimize overlaps
    adjust_text(texts, autoalign='xy', ha='right', va='bottom', only_move={'points': 'y', 'text': 'xy'})

    legend_lines = [Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='green', lw=4)]

    plt.legend(legend_lines, ['Low speed tracks', 'Medium speed tracks', 'High speed tracks'],
               loc='upper right', bbox_to_anchor=(1, 0.85), fontsize='x-large')

    ax.axis('off')
    ax.grid(False)
    plt.title('SIMILARITY BETWEEN CIRCUITS', font='Fira Sans', fontsize=28)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig('../PNGs/Track clusters.png', dpi=400)
    plt.show()


def dhl_pitstops(year, round=None, exclude=None, points=False):
    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    pitstops = pd.read_csv('../resources/Pit stops.csv', sep='|')
    pitstops = pitstops[pitstops['Year'] == year]
    colors = []
    if round is None:
        if points:
            pitstops = pitstops.groupby('Team')['Points'].sum()
        else:
            pitstops = pitstops.groupby('Driver')['Time'].mean()
    else:
        pitstops = pitstops[pitstops['Race_ID'] == round]
    pitstops = pitstops.reset_index()
    if points:
        pitstops = pitstops.sort_values(by='Points', ascending=False)
        color_data = [i for i in pitstops['Team']]
        for c_data in color_data:
            colors.append(team_colors_2023[c_data])
        plot_size = (10, 9)
        annotate_fontsize = 16
        y_offset_rounded = -10
        y_offset_annotate = 1
        title = 'DHL PIT STOPS POINTS'
        y_label = 'Points'
        x_label = 'Teams'
    else:
        pitstops = pitstops.sort_values(by='Time', ascending=True)
        pitstops['Time'] = pitstops['Time'].round(2)
        drivers = [i for i in pitstops['Driver']]
        plot_size = (17, 10)
        annotate_fontsize = 12
        y_offset_rounded = 0
        y_offset_annotate = 0.05
        title = 'PIT STOP TIMES'
        y_label = 'Time (s)'
        x_label = 'Driver'

        for driver in drivers:
            for key, value in fastf1.plotting.DRIVER_COLORS.items():
                parts = key.split(" ", 1)
                new_key = parts[1] if len(parts) > 1 else key
                if (new_key == driver.lower()) or (new_key == 'guanyu' and driver == 'Zhou'):
                    colors.append(value)
                    break

    fig, ax1 = plt.subplots(figsize=plot_size)

    if round is not None:
        name_count = {}

        def update_name(name):
            if name in name_count:
                name_count[name] += 1
            else:
                name_count[name] = 1
            return f"{name} {name_count[name]}"

        pitstops['Driver'] = pitstops['Driver'].apply(update_name)

    if exclude is not None:
        pitstops = pitstops[~pitstops['Driver'].isin(exclude)]

    if points:
        bars = ax1.bar(pitstops['Team'], pitstops['Points'], color=colors,
                       edgecolor='white')
    else:
        bars = ax1.bar(pitstops['Driver'], pitstops['Time'], color=colors,
                       edgecolor='white')

    round_bars(bars, ax1, colors, y_offset_rounded)
    annotate_bars(bars, ax1, y_offset_annotate, annotate_fontsize,  ceil_values=True)

    ax1.set_title(title, font='Fira Sans', fontsize=28)
    ax1.set_xlabel(x_label, font='Fira Sans', fontsize=20)
    ax1.set_ylabel(y_label, font='Fira Sans', fontweight='bold', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=18)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(False)
    font_properties = {'family': 'Fira Sans', 'size': 14}

    # Set x-ticks and y-ticks font
    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_properties)

    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
    plt.xticks(rotation=90)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim([1.6, ymax])
    plt.tight_layout()
    plt.savefig(f'../PNGs/PIT STOP AVG TIME {year}', dpi=400)
    plt.show()


def get_retirements_per_driver(driver, start=None, end=None):
    ergast = Ergast()
    positions = pd.Series(dtype=object)

    for i in range(start, end):
        races = ergast.get_race_results(season=i, limit=1000)
        total_races = races.content
        for race in total_races:
            race = race[race['familyName'] == driver]
            if not pd.isna(race['status'].max()):
                if re.search(r'(Spun off|Accident|Collision)', race['status'].max()):
                    positions = pd.concat([positions, pd.Series(['Accident DNF'])], ignore_index=True)
                elif re.search(r'(Finished|\+)', race['status'].max()):
                    positions = pd.concat([positions, pd.Series(['P' + str(race['position'].max())])],
                                          ignore_index=True)
                else:
                    positions = pd.concat([positions, pd.Series(['Mechanical DNF'])], ignore_index=True)
        print(i)

    positions = positions.value_counts()
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

    def func(pct, allvalues):
        absolute = int(round(pct / 100. * np.sum(allvalues), 2))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    top_N.plot.pie(ax=ax, autopct=lambda pct: func(pct, top_N), labels=['' for _ in top_N.index], legend=False,
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, colors=colors,
                   textprops={"color": "black"})  # Set line color to black

    ax.legend(title="Finish legend", loc="center left", labels=top_N.index, bbox_to_anchor=(0.8, 0.1))

    plt.title(f'{driver} finish history (Total races: {positions.sum()})',
              font='Fira Sans', fontsize=16, color='white')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver} finish history', dpi=150)
    plt.show()
    print(positions)


def compare_drivers_season(d_1, d_2, season, DNFs=False):
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
        driver = qualy[qualy['position'] == best_pos]['driverCode'].min()
        qualy_result.append(driver)

    print(Counter(race_result))
    print(f'QUALYS: {Counter(qualy_result)}')
    print(f'{d_1} points: {d1_points}')
    print(f'{d_2} points: {d2_points}')


def get_pit_stops_ergast(year):
    ergast = Ergast()
    schedule = ergast.get_race_schedule(season=year, limit=1000)
    circuits = schedule.circuitId.values
    circuits = [item.replace("_", " ").title() for item in circuits]

    n_pit_stops = []
    if year == 2023:
        pitstops = pd.read_csv('../resources/Pit stops.csv', sep='|')
        pitstops = pitstops[pitstops['Year'] == year]
        pitstops = pitstops.groupby('Race_ID')['Pos'].max()
        pitstops = pitstops.to_frame()
        circuits = [circuits[i] for i in range(len(pitstops))]
        pitstops['Circuits'] = circuits
        n_pit_stops = np.array(pitstops['Pos'])

    else:
        for i in range(len(schedule)):
            pit_stop = ergast.get_pit_stops(season=year, round=i + 1, limit=1000)
            n_pit_stops.append(len(pit_stop.content[0]))

    fig, ax1 = plt.subplots(figsize=(12, 8))

    total = sum(n_pit_stops)
    mean = round(total / len(n_pit_stops), 2)
    bars = ax1.bar(circuits, n_pit_stops, color="#AED6F1", edgecolor='white', label='Total Pitstops')
    ax1.set_title(f'PIT STOPS IN {year} (Total: {total} - Avg: {mean})', font='Fira Sans', fontsize=24)
    ax1.set_xlabel('Circuit', font='Fira Sans', fontsize=16)
    ax1.set_ylabel('Total Pitstops', font='Fira Sans', fontsize=16)
    ax1.yaxis.grid(False)
    ax1.xaxis.grid(False)

    round_bars(bars, ax1, '#AED6F1', color_1=None, color_2=None, y_offset_rounded=0, corner_radius=0.3)
    annotate_bars(bars, ax1, 0.5, 14, text_annotate='default', ceil_values=False)

    ax2 = ax1.twinx()
    mean_pits = [round(i / 20, 2) for i in n_pit_stops]

    total = sum(mean_pits)
    mean = round(total / len(mean_pits), 2)

    ax2.plot(circuits, mean_pits, color='red', linestyle='dashed', linewidth=2, markersize=8,
             label='Avg PitStops per driver')
    ax2.set_ylabel(f'PitStops per driver (Avg per race: {mean})', fontweight='bold')
    ax2.yaxis.grid(True, linestyle='dashed')

    for i, value in enumerate(mean_pits):
        ax2.annotate(f'{value}', (i, value), textcoords="offset points", xytext=(0, -12), ha='center',
                     fontsize=12, bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='black'))

    # Add a legend
    rounded_bar_legend = mpatches.Patch(facecolor='#AED6F1', label='Total Pitstops')
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend([rounded_bar_legend] + handles2, ['Total Pitstops'] + labels2, loc='upper left')

    ax1.set_xticklabels(circuits, rotation=90)
    plt.tight_layout()
    plt.savefig(f'../PNGs/PIT STOPS IN {year}.png', dpi=400)
    plt.show()


def get_retirements():
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
    end = 0
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
    ergast = Ergast()
    circuitos = []

    for i in range(1950, 2024):
        season_circuits = ergast.get_circuits(season=i, limit=10000)
        circuitos.append(season_circuits)

    circuit_names = []
    for season in circuitos:
        circuit_names.extend(season['country'])
    series = pd.Series(circuit_names)

    series[series == 'United States'] = 'USA'
    series = series.value_counts()
    N = 12
    top_N = series.nlargest(N)
    top_N['Other'] = series.iloc[N:].sum()

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

    ax.legend(title="Country", loc="center left", labels=top_N.index, bbox_to_anchor=(0.8, 0.1))

    plt.title(f'Times race in a country (Total races: {total_value})', fontsize=16, color='white')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'../PNGs/Countrys in F1 History', dpi=800)
    plt.show()

    items_per_table = 20
    start = 1
    end = 1
    chunks = np.array_split(series, np.ceil(len(series) / items_per_table))

    for i in range(len(chunks)):
        other_values = pd.DataFrame(chunks[i]).reset_index().rename(columns={0: 'Total times',
                                                                             'index': 'Country'})
        end = start + len(other_values)
        other_values['Ranking'] = range(start, end)
        other_values = other_values[['Ranking', 'Country', 'Total times']]

        render_mpl_table(other_values, diff_column='No', col_Widths=[0.1, 0.3, 0.15],
                         title=f'TOP {start} - {end - 1} countries')

        start = end
        plt.tight_layout()
        plt.savefig(f'../PNGs/TOP {start} - {end - 1} countries', dpi=600)
        plt.show()


def get_topspeed(gp):
    top_speed_array = []

    for i in range(gp):

        session = fastf1.get_session(2023, i + 1, 'Q')
        session.load(telemetry=True, weather=False)
        circuit_speed = {}

        for lap in session.laps.pick_quicklaps().iterrows():
            top_speed = max(lap[1].telemetry['Speed'])
            driver = lap[1]['Driver']
            driver_speed = circuit_speed.get(driver)
            if driver_speed is not None:
                if top_speed > driver_speed:
                    circuit_speed[driver] = top_speed
            else:
                circuit_speed[driver] = top_speed

            print(circuit_speed)

        max_key = max(circuit_speed, key=circuit_speed.get)
        driver_top_speed = f'{max_key} - {circuit_speed[max_key]} - {session.event["EventName"]}'

        top_speed_array.append(driver_top_speed)

        print(top_speed_array)

    print(top_speed_array)


def get_fastest_data(session, column='Speed', fastest_lap=None, DRS=True):
    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    circuit_speed = {}
    colors_dict = {}

    if fastest_lap is not None:
        drivers = session.laps['Driver'].groupby(session.laps['Driver']).size()

        drivers = list(drivers.reset_index(name='Count')['Driver'].values)

        for driver in drivers:
            lap = session.laps.pick_driver(driver).pick_fastest()
            if lap.Team is not np.nan:
                top_speed = max(lap.telemetry[column])
                driver_speed = circuit_speed.get(driver)
                team = lap.Team.lower()
                if team == 'red bull racing':
                    team = 'red bull'
                elif team == 'haas f1 team':
                    team = 'haas'
                if driver_speed is not None:
                    if top_speed > driver_speed and column == 'Speed':
                        circuit_speed[driver] = top_speed
                        colors_dict[driver] = team
                    elif top_speed < driver_speed and column != 'Speed':
                        circuit_speed[driver] = top_speed
                        colors_dict[driver] = team
                else:
                    circuit_speed[driver] = top_speed
                    colors_dict[driver] = team

            print(circuit_speed)
    else:
        laps = session.laps.pick_quicklaps()

        for lap in laps.iterrows():
            try:
                if column == 'Speed':
                    if lap[1].telemetry['DRS'].max() >= 10 and not DRS:
                        top_speed = 0
                    elif column == 'Speed':
                        top_speed = max(lap[1].telemetry[column])
                else:
                    top_speed = round(lap[1][column].total_seconds(), 3)
            except ValueError as e:
                continue
            driver = lap[1]['Driver']
            driver_speed = circuit_speed.get(driver)
            team = lap[1].Team.lower()
            if team == 'red bull racing':
                team = 'red bull'
            elif team == 'haas f1 team':
                team = 'haas'
            if driver_speed is not None:
                if top_speed > driver_speed and column == 'Speed':
                    circuit_speed[driver] = top_speed
                    colors_dict[driver] = team
                elif top_speed < driver_speed and column != 'Speed':
                    circuit_speed[driver] = top_speed
                    colors_dict[driver] = team
            else:
                circuit_speed[driver] = top_speed
                colors_dict[driver] = team

            print(circuit_speed)

    if column == 'Speed':
        order = True
        if fastest_lap is not None:
            column = 'Top Speeds (only the fastest lap from each driver)'
        else:
            column = 'Top Speeds'
        x_fix = 5
        y_fix = 0.25
    else:
        y_fix = 0.025
        x_fix = 0.75
        order = False
        column = f"{column[:-5]} {column[-5:-4]} Times"
        if not DRS:
            column += 'without DRS'

    circuit_speed = {k: v for k, v in sorted(circuit_speed.items(), key=lambda item: item[1], reverse=order)}

    fig, ax1 = plt.subplots(figsize=(20, 8))

    colors = []
    for i in range(len(circuit_speed)):
        colors.append(fastf1.plotting.TEAM_COLORS[colors_dict[list(circuit_speed.keys())[i]]])

    bars = ax1.bar(list(circuit_speed.keys()), list(circuit_speed.values()), color=colors,
                   edgecolor='white')

    round_bars(bars, ax1, colors, y_offset_rounded=0)
    annotate_bars(bars, ax1, y_fix, 14, text_annotate='default', ceil_values=False)

    ax1.set_title(f'{column} in {str(session.event.year) + " " + session.event.Country + " " + session.name}',
                  font='Fira Sans', fontsize=28)
    ax1.set_xlabel('Driver', fontweight='bold', fontsize=12)
    if 'Speed' in column:
        y_label = 'Max speed'
    else:
        y_label = 'Sector time'
    ax1.set_ylabel(y_label, fontweight='bold', fontsize=12)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(False)

    max_value = max(circuit_speed.values())

    # Adjust the y-axis limits
    ax1.set_ylim(min(circuit_speed.values()) - x_fix, max_value + x_fix)

    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)

    def format_ticks(val, pos):
        return '{:.0f}'.format(val)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{column} IN {str(session.event.year) + " " + session.event.Country + " " + session.name}',
                dpi=400)
    plt.show()


def wins_and_poles_circuit(circuit, start=None, end=None):
    ergast = Ergast()
    winners = pd.Series(dtype=str)
    poles = pd.Series(dtype=str)
    years = []
    if end is None:
        end = 2024
    if start is None:
        start = 1950
    for i in range(start, end):
        races = ergast.get_race_results(season=i, limit=1000)
        has_raced = races.description[races.description['circuitId'] == circuit]
        if len(has_raced) > 0:
            index = has_raced['circuitId'].index[0]
            actual_race = races.content[index]
            data = actual_race[actual_race['position'] == 1]
            qualy_data = actual_race[actual_race['grid'] == 1]
            driver_win = str(data['givenName'].min() + ' ' + data['familyName'].min())
            driver_pole = str(qualy_data['givenName'].min() + ' ' + qualy_data['familyName'].min())
            winners = pd.concat([winners, pd.Series([driver_win])], ignore_index=True)
            poles = pd.concat([poles, pd.Series([driver_pole])], ignore_index=True)
            years.append(i)
        print(years)

    winners.index = years
    poles.index = years

    winners = (winners.groupby(winners).agg(["count", lambda x: list(x.index)])
               .rename(columns={"count": "Times", "<lambda_0>": "Years"}))

    poles = (poles.groupby(poles).agg(["count", lambda x: list(x.index)])
             .rename(columns={"count": "Times", "<lambda_0>": "Years"}))

    winners = winners.sort_values("Times", ascending=False)
    poles = poles.sort_values("Times", ascending=False)

    print('POLES')
    print(poles)
    print('WINNERS')
    print(winners)


def get_historical_race_days():
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
    with open('../resources/Races by day.txt', 'w') as file:
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
    with open('../resources/Grouped races by day.txt', 'w') as file:
        for key, value in sorted_days:
            file.write(f'{key}: {value}\n')


def plot_overtakes():
    df = pd.read_csv('../resources/Overtakes.csv')
    df = df[df['Season'] >= 1999]
    df = df[~df['Race'].str.contains('Sprint')]
    df = df[~df['Race'].str.contains('Season')]

    df = (df.groupby('Season').agg({'Season': 'count', 'Overtakes': 'sum'})
          .rename(columns={'Season': 'Races'}))

    years = df.index.values

    fig, ax1 = plt.subplots(figsize=(24, 10))

    bars = ax1.bar(years, df['Overtakes'], color="#AED6F1", edgecolor='white')
    ax1.set_title(f'OVERTAKES IN F1 HISTORY', fontsize=24)
    ax1.set_xlabel('YEAR', fontweight='bold')
    ax1.set_ylabel('TOTAL OVERTAKES', fontweight='bold')
    ax1.yaxis.grid(False)
    ax1.xaxis.grid(False)

    for bar in bars:
        height = bar.get_height()
        x_value = bar.get_x() + bar.get_width() / 2

        # Customize color and label based on the x-value
        if x_value < 2010:
            color = 'orange'
        elif x_value < 2023:
            color = 'green'
        else:
            color = 'green'
        if x_value == 2005:
            color = 'red'
        if x_value == 2010:
            color = 'yellow'

        bar.set_color(color)
        ax1.text(x_value, height + 10, f'{height}', ha='center', va='bottom', fontsize=10, zorder=100)

    # Create custom legend entries
    legend_entries = [mpatches.Patch(color='orange', label='WITH refueling'),
                      mpatches.Patch(color='red', label='WITH refueling, NO TYRE CHANGES'),
                      mpatches.Patch(color='yellow', label='NO refueling'),
                      mpatches.Patch(color='green', label='NO refueling, WITH DRS')]

    ax2 = ax1.twinx()
    mean_overtakes = round(df['Overtakes'] / df['Races'], 2)
    ax2.axhline(y=round(mean_overtakes.mean(), 2), color='red', linestyle='--',
                label='Avg overtakes per race since 1999', zorder=-1000)
    ax2.plot(years, mean_overtakes, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=8,
             label='Avg overtakes per race in that season')
    ax2.set_ylabel(f'Overtakes per season (Avg per season: {round(mean_overtakes.mean(), 2)})', fontweight='bold')
    ax2.yaxis.grid(True, linestyle='dashed')

    for year, value in mean_overtakes.items():
        ax2.annotate(f'{value}', (year, value), textcoords="offset points", xytext=(0, -20), ha='center',
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white'))

    # Add a legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2 + legend_entries,
               labels1 + labels2 + [entry.get_label() for entry in legend_entries], loc='upper right')
    plt.xticks(years, rotation=45)

    plt.tight_layout()
    plt.savefig(f'../PNGs/OVERTAKES IN F1.png', dpi=400)

    plt.show()


def races_by_driver_dorsal(number):
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


def plot_circuit():
    session = fastf1.get_session(2022, 'Austin', 'Q')
    session.load()

    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()

    circuit_info = session.get_circuit_info()

    def rotate(xy, *, angle):
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)]])
        return np.matmul(xy, rot_mat)

    track = pos.loc[:, ('X', 'Y')].to_numpy()

    # Also get the elevation data
    elevation = pos.loc[:, 'Z'].to_numpy()

    # Convert the rotation angle from degrees to radian.
    track_angle = circuit_info.rotation / 180 * np.pi

    # Rotate the track map.
    rotated_track = rotate(track, angle=track_angle)

    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    # Normalize X, Y coordinates
    norm_track = np.zeros_like(rotated_track)
    norm_track[:, 0] = normalize(rotated_track[:, 0])
    norm_track[:, 1] = normalize(rotated_track[:, 1])
    norm_elevation = normalize(elevation)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.set_zlim(0, 1)

    # Vertical offset for each layer
    vertical_offset = 0.125  # Adjust this based on how much separation you want between layers

    # Base elevation for all layers, set to zero
    base_elevation = np.zeros_like(norm_elevation)

    # Loop through layers (replace this loop to use actual multiple laps or segments if available)

    # Start each layer from Z=0 and then build it up by the layer number multiplied by the vertical offset
    current_elevation = base_elevation + norm_elevation * vertical_offset

    # Plot the track in 3D
    ax.plot(norm_track[:, 0], norm_track[:, 1], current_elevation)

    verts = []
    for j in range(len(norm_track) - 1):
        x = norm_track[j:j + 2, 0]
        y = norm_track[j:j + 2, 1]
        z = current_elevation[j:j + 2]
        poly = [
            [x[0], y[0], 0],
            [x[1], y[1], 0],
            [x[1], y[1], z[1]],
            [x[0], y[0], z[0]]
        ]
        verts.append(poly)
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors='b'))

    # Turn off grid
    ax.grid(False)

    # Turn off axis values and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Hide axes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()

    plt.show()

    '''
    def find_closest_point(track, target_x, target_y):
        distances = np.sqrt((track[:, 0] - target_x) ** 2 + (track[:, 1] - target_y) ** 2)
        closest_index = np.argmin(distances)
        return closest_index, track[closest_index]
    # Iterate over all corners.
    for _, corner in circuit_info.corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner['Angle'] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

        # Add the offset to the position of the corner
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y

        # Rotate the text position equivalently to the rest of the track map
        track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

        # Find the closest point in the rotated_track to (track_x, track_y)
        closest_index, closest_point = find_closest_point(rotated_track, track_x, track_y)

        # Now, closest_index is the index of the closest point in rotated_track
        # closest_point is the [x, y] of the closest point

        # If you also need the Z coordinate, you can get it from the original 'track' array
        track_z = track[closest_index, 2]

        # Draw a circle next to the track in 3D.
        ax.scatter(text_x, text_y, track_z, color='grey', s=140)

        # Draw a line from the track to this circle in 3D.
        ax.plot([track_x, text_x], [track_y, text_y], [track_z, track_z], color='grey')

        # Finally, print the corner number inside the circle.
        # NOTE: text plotting in 3D isn't as straightforward as in 2D
        ax.text(text_x, text_y, track_z, txt, color='white')
        
    '''


def avg_driver_position(driver, team, year, session='Q'):
    ergast = Ergast()
    if session == 'Q':
        data = ergast.get_qualifying_results(season=year, limit=1000)
    else:
        data = ergast.get_race_results(season=year, limit=1000)

    position = []

    if driver is None:
        drivers = []
        for gp in data.content:
            for d in gp['driverCode']:
                drivers.append(d)
        drivers_array = set(drivers)
        drivers = {d: [] for d in drivers_array}
        for gp in data.content:
            for d in drivers_array:
                data = gp[gp['driverCode'] == d]
                if len(data) > 0:
                    pos = data['position'].values[0]
                    drivers[d].append(pos)
        avg_grid = {}
        for key, pos_array in drivers.items():
            mean_pos = round(np.mean(pos_array), 2)
            avg_grid[key] = mean_pos
        avg_grid = dict(sorted(avg_grid.items(), key=lambda item: item[1]))
        drivers = list(avg_grid.keys())
        avg_pos = list(avg_grid.values())
        colors = [driver_colors_2023[key] for key in drivers]
        fig, ax = plt.subplots(figsize=(12, 6))  # Set the figure size (optional)
        bars = plt.bar(drivers, avg_pos, color=colors)  # Plot the bar chart with specific colors (optional)

        for bar in bars:
            bar.set_visible(False)

        i = 0
        for bar in bars:
            height = bar.get_height()
            x, y = bar.get_xy()
            width = bar.get_width()

            # Create a fancy bbox with rounded corners and add it to the axes
            rounded_box = rounded_top_rect(x, y, width, height, 0.1, colors[i], -0.1)
            rounded_box.set_facecolor(colors[i])
            ax.add_patch(rounded_box)
            i += 1

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f'{height}', ha='center',
                     va='bottom',
                     font='Fira Sans', fontsize=11)
        plt.xlabel('Drivers', font='Fira Sans', fontsize=14)  # x-axis label (optional)
        plt.ylabel('Avg Grid Position', font='Fira Sans', fontsize=14)  # y-axis label (optional)
        plt.title('Average Grid Position Per Driver', font='Fira Sans', fontsize=20)  # Title (optional)
        plt.xticks(rotation=90, fontsize=13)
        plt.yticks(fontsize=13)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'../PNGs/Average grid position {year}.png', dpi=400)
        plt.show()
    else:
        for gp in data.content:
            session_data = gp[(gp['driverId'] == driver) & (gp['constructorId'] == team)]
            if len(session_data) > 0:
                position.append(session_data['position'].values[0])
            else:
                print(f'{driver} not in {team}')

    print(np.round(np.mean(position), 2))
    return np.round(np.mean(position), 2), statistics.median(position)


def lucky_drivers(start=None, end=None):
    if start is None:
        start = 1950
    if end is None:
        end = 2024
    ergast = Ergast()
    drivers_array = []
    all_races = []
    for i in range(start, end):
        races = ergast.get_race_results(season=i, limit=1000).content
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
                teams = set(race_data['constructorId'])
                for team in teams:
                    teammate = race[race['constructorId'] == team]
                    teammate = list(teammate['givenName'] + ' ' + teammate['familyName'].values)
                    if len(teammate) > 1:
                        loops = teammate.count(driver)
                        teammate = [x for x in teammate if x != driver]
                        teammate = set(teammate)
                        for i in range(loops):
                            for team_name in teammate:
                                teammate_data = race[(race['givenName'] == team_name.split(' ')[0]) & (
                                        race['familyName'] == team_name.split(' ', 1)[1])]
                                teammate_data = teammate_data[teammate_data['constructorId'] == team]
                                status_d1 = race_data[race_data['constructorId'] == team]['status'].values[i]
                                if len(teammate_data) > 0:
                                    for j in range(len(teammate_data)):
                                        status_d2 = teammate_data['status'].values[j]
                                        if re.search(r'(Spun off|Accident|Withdrew|Collision|Finished|Did|\+)',
                                                     status_d1):
                                            if not re.search(r'(Spun off|Accident|Withdrew|Collision|Finished|Did|\+)',
                                                             status_d2):
                                                luck[driver] += 1
                                        else:
                                            if re.search(r'(Spun off|Accident|Withdrew|Collision|Finished|Did|\+)',
                                                         status_d2):
                                                luck[driver] -= 1
        print(driver)

    # Sort dictionary by its values in ascending order
    sorted_data = {k: v for k, v in sorted(luck.items(), key=lambda item: item[1], reverse=False)}

    values = 0
    for key, value in sorted_data.items():
        print(f"{key}: {value}")
        values += value
    print(values)


def get_fastest_punctuable_lap(circuit, start=None, end=None, all_drivers=False):
    if start is None:
        start = 1950
    if end is None:
        end = 2024

    current_fl = pd.Timedelta(days=1)
    current_driver = ''
    current_year = 0

    ergast = Ergast()
    for year in range(start, end):
        round_number = ergast.get_race_schedule(season=year, circuit=circuit, limit=1000)
        if len(round_number):
            round_number = round_number.values[0][1]
            laps = ergast.get_lap_times(season=year, round=round_number, limit=1000)
            if len(laps.content) > 0:
                race_results = ergast.get_race_results(season=year, round=round_number, limit=1000)
                if not all_drivers:
                    top_10 = race_results.content[0]['driverId'][:10].values
                    fl_drivers = laps.content[0][laps.content[0]['driverId'].isin(top_10)]
                    fl = fl_drivers['time'].min()
                    driver = fl_drivers[fl_drivers['time'] == fl]['driverId'].values[0]
                else:
                    fl = laps.content[0]['time'].min()
                    driver = laps.content[0][laps.content[0]['time'] == fl]['driverId'].values[0]

                if fl < current_fl:
                    current_fl = fl
                    current_driver = driver
                    current_year = year
                print(current_fl, current_driver, current_year)


def get_driver_results_circuit(driver, circuit, start=None, end=None):
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
