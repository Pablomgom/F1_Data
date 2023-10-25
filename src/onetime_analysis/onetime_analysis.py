import fastf1
import numpy as np
import pandas as pd
import re
import statistics
from unidecode import unidecode
from adjustText import adjust_text
from fastf1 import plotting
from fastf1.ergast import Ergast
from matplotlib import cm, ticker
from collections import Counter
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from statsmodels.tsa.arima.model import ARIMA

from src.ergast_api.my_ergast import My_Ergast
from src.plots.plots import annotate_bars, title_and_labels, get_handels_labels, get_font_properties
from src.general_analysis.table import render_mpl_table

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from src.plots.plots import round_bars
from src.utils.utils import append_duplicate_number, update_name, get_quartiles
from src.variables.variables import team_colors_2023, driver_colors_2023, point_system_2010, point_system_2009, \
    point_systems, driver_colors_historical


def get_DNFs_team(team, start, end):
    """
         Print the DNFs of a team

         Parameters:
         team (str): Team
         start (int): Year of start
         end (int): Year of end
    """

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


def proccess_season_data(data, drivers, driver_points, system):
    """
        Print the points for a season

        Parameters:
        data (DataFrame): Data
        drivers (int): Drivers to analyze
        driver_points (dict): Dict with the driver's points
        system (dict): Point system
   """

    def print_wdc(driver_points):
        driver_points = dict(sorted(driver_points.items(), key=lambda item: item[1], reverse=True))
        total_p = 0
        pos = 1
        for d, p in driver_points.items():
            print(f'{pos}: {d} - {p}')
            total_p += p
            pos += 1
        print(total_p)

    for i in range(len(data)):
        for driver in drivers:
            driver_data = data[i][data[i]['driverCode'] == driver]
            if len(driver_data) > 0:
                pos = data[i][data[i]['driverCode'] == driver]['position'].values[0]
                if pos in list(system.keys()):
                    driver_points[driver] += system[pos]
        if i == len(data) - 2:
            print_wdc(driver_points)

    print_wdc(driver_points)


def simulate_season_different_psystem(year, system):
    """
       Simulate a season with another point system

        Parameters:
        year (int): Data
        system (dict): Point system
   """

    ergast = Ergast()
    race_data = ergast.get_race_results(season=year, limit=1000).content
    drivers = set([code for df in race_data for code in df['driverCode'].values])
    driver_points = {}
    for driver in drivers:
        driver_points[driver] = 0
    system = point_systems[system]
    proccess_season_data(race_data, drivers, driver_points, system)


def simulate_qualy_championship(year, system):
    """
       Simulate a qualy WDC with a point system

        Parameters:
        year (int): Data
        system (dict): Point system
   """

    qualy_data = Ergast().get_qualifying_results(season=year, limit=1000).content
    drivers = set([code for df in qualy_data for code in df['driverCode'].values])
    driver_points = {}
    for driver in drivers:
        driver_points[driver] = 0
    system = point_systems[system]
    proccess_season_data(qualy_data, drivers, driver_points, system)


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
    """
       Compare qualy results for a team given a threshold

        Parameters:
        team (str, optional): Only for analyze a team. Default: None
        threshold (int): Threshold to compare the values
        end (int, optional): 1950 by default
        exclude (dict, optional): Excludes a circuit in a given year

   """

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
    """
       Compare points for a team given a threshold

        Parameters:
        team (str, optional): Only for analyze a team. Default: None
        threshold (int): Threshold to compare the values
        end (int, optional): 1950 by default
        exclude (dict, optional): Excludes a circuit in a given year

   """

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
    """
       Compare points for a team given a threshold

        Parameters:
        year (int): Year to plot
        session (str, optional): Qualy or Race. Default. Q
        predict (bool, optional): Predicts outcome of the season. Default: False
        mode (bool, optional): Total sum or 4 MA. Default: None(4 MA)

   """
    reverse = True
    ergast = Ergast()
    data = ergast.get_race_results(season=year, limit=1000)
    teams = set(data.content[0]['constructorName'])
    team_points = pd.DataFrame(columns=['Team', 'Points', 'Circuit'])
    yticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    if session == 'Q':
        data = ergast.get_qualifying_results(season=year, limit=1000)
        yticks = [1, 3, 6, 9, 12, 15, 18, 20]
        title = 'Average qualy position in the last 4 GPs'
        reverse = False
    else:
        if predict:
            title = 'Average points prediction in the last 4 GPs'
        else:
            title = 'Average points in the last 4 GPs'

    circuits = np.array(data.description['circuitId'])
    circuits = [i.replace('_', ' ').title() for i in circuits]
    circuits = append_duplicate_number(circuits)
    for i in range(len(data.content)):
        for team in teams:
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

    ax = transposed.plot(figsize=(12, 12), marker='o', color=ordered_colors, markersize=7, lw=3)

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
    last_values = transposed.iloc[-1].values
    handles, labels = ax.get_legend_handles_labels()
    colors = [line.get_color() for line in ax.lines]
    info = list(zip(handles, labels, colors, last_values))
    info.sort(key=lambda item: item[3], reverse=reverse)
    handles, labels, colors, last_values = zip(*info)
    labels = [f"{label} ({last_value:.2f})" for label, last_value in zip(labels, last_values)]

    if predict:
        handles = list(handles)
        handles.append(predict_patch)
        labels.append("Predictions")

    plt.legend(handles=handles, labels=labels, prop=font, loc="upper left", bbox_to_anchor=(1, 0.6))

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
    plt.savefig(f'../PNGs/AVERAGE POINTS.png', dpi=450)
    plt.show()


def plot_upgrades(scope=None):
    """
       Plot the upgrades for a season

        Parameters:
        scope (str, optional): (Performance|Circuit Specific)

   """

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
    last_values = transposed.iloc[-1].values
    font = get_font_properties('Fira Sans', 12)

    if scope is None:
        scope = ''
    else:
        scope += ' '

    ax = transposed.plot(figsize=(10, 12), marker='o', color=ordered_colors, markersize=7, lw=3)

    title_and_labels(plt, f'Cumulative {scope}Upgrades for Each Team', 28,
                     'Races', 18, 'Number of Upgrades', 18, 0.5)

    handles, labels = get_handels_labels(ax)
    colors = [line.get_color() for line in ax.lines]
    info = list(zip(handles, labels, colors, last_values))
    info.sort(key=lambda item: item[3], reverse=True)
    handles, labels, colors, last_values = zip(*info)
    labels = [f"{label} ({last_value:.0f})" for label, last_value in zip(labels, last_values)]

    plt.legend(handles=handles, labels=labels, prop=font, loc="upper left", fontsize='x-large')
    plt.xticks(ticks=range(len(transposed)), labels=transposed.index,
               rotation=90, fontsize=12, fontname='Fira Sans')
    plt.yticks(fontsize=12, fontname='Fira Sans')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f'../PNGs/{scope} UPGRADES.png', dpi=400)
    plt.show()


def cluster_circuits(year, rounds, prev_year=None, circuit=None, clusters=None):
    """
        Cluster circuits

        Parameters:
        year (int): Year to plot
        rounds (int): Rounds to analyze
        prev_year (int, optional): Take data from a session of the previous year. Default: None
        circuit (str, optional): Circuit to take data from last year. Default: None
        clusters(int, optional): Number of clusters

   """

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

        q1, median, q3, avg = get_quartiles(sorted_speed)

        full_gas = telemetry[telemetry['Throttle'].isin([100, 99])]
        full_gas = len(full_gas) / len(telemetry)
        lifting = telemetry[(telemetry['Throttle'] >= 1) & (telemetry['Throttle'] <= 99)]
        lifting = len(lifting) / len(telemetry)
        data.append([corners_per_meter, max_speed, q1, median, q3, avg, full_gas, lifting])

        circuits.append(session.event.Location)

    data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(data)

    cluster_means = {}
    for cluster_id in np.unique(y_kmeans):
        cluster_means[cluster_id] = data[y_kmeans == cluster_id].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    texts = []
    colors = ['red', 'blue', 'green']  # Assuming 3 clusters; extend this list if there are more clusters

    for i, name in enumerate(circuits):
        ax.scatter(principal_components[i, 0], principal_components[i, 1], color=colors[y_kmeans[i]], s=100)
        texts.append(ax.text(principal_components[i, 0], principal_components[i, 1], name,
                             font='Fira Sans', fontsize=13))

    for i, center in enumerate(pca.transform(kmeans.cluster_centers_)):
        ax.scatter(center[0], center[1], s=300, c='#FF8C00')

    adjust_text(texts, autoalign='xy', ha='right', va='bottom', only_move={'points': 'y', 'text': 'xy'})

    legend_lines = [Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='green', lw=4)]

    plt.legend(legend_lines, ['Low speed tracks', 'Medium speed tracks', 'High speed tracks'],
               loc='lower center', fontsize='large')

    ax.axis('off')
    ax.grid(False)
    plt.title('SIMILARITY BETWEEN CIRCUITS', font='Fira Sans', fontsize=28)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig('../PNGs/Track clusters.png', dpi=450)
    plt.show()


def dhl_pitstops(year, groupBy='Driver', round=None, exclude=None, points=False):
    """
        Print pitstops given the dhl data

        Parameters:
        year (int): Year to plot
        groupBy (str): Driver or Team. Default: Driver
        round (int, optional): Plot only a given round. Default: None
        exclude (list, optional): Exclude pit stops. Default: None
        points(bool, optional): Plot DHL system points. Default: False

   """

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    pitstops = pd.read_csv('../resources/Pit stops.csv', sep='|')
    pitstops = pitstops[pitstops['Year'] == year]
    colors = []
    if round is None:
        if points:
            max_round = pitstops['Race_ID'].max()
            print(
                pitstops[pitstops['Race_ID'] == max_round].groupby('Team')['Points'].sum().sort_values(ascending=False))
            pitstops = pitstops.groupby('Team')['Points'].sum()
        else:
            pitstops = pitstops.groupby(groupBy)['Time'].median()
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
        drivers = [i for i in pitstops[groupBy]]
        if groupBy == 'Driver':
            plot_size = (17, 10)
            x_label = 'Driver'
        else:
            plot_size = (12, 10)
            x_label = 'Team'
        annotate_fontsize = 12
        y_offset_rounded = 0
        y_offset_annotate = 0.05
        title = 'MEDIAN PIT STOP TIMES'
        if round is not None:
            title = 'PIT STOPS TIME'
        y_label = 'Time (s)'

        for driver in drivers:
            for key, value in fastf1.plotting.DRIVER_COLORS.items():
                parts = key.split(" ", 1)
                new_key = parts[1] if len(parts) > 1 else key
                if (new_key == driver.lower()) or (new_key == 'guanyu' and driver == 'Zhou'):
                    colors.append(value)
                    break

    fig, ax1 = plt.subplots(figsize=plot_size)

    if round is not None and groupBy == 'Driver':
        pitstops['Driver'] = pitstops['Driver'].apply(update_name)

    if exclude is not None:
        pitstops = pitstops[~pitstops['Driver'].isin(exclude)]

    if points:
        bars = ax1.bar(pitstops['Team'], pitstops['Points'], color=colors,
                       edgecolor='white')
    else:
        bars = ax1.bar(pitstops[groupBy], pitstops['Time'], color=colors,
                       edgecolor='white')

    if groupBy == 'Team':
        colors = [team_colors_2023[i] for i in pitstops['Team'].values]
        annotate_fontsize = 20
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    round_bars(bars, ax1, colors, y_offset_rounded=y_offset_rounded)
    annotate_bars(bars, ax1, y_offset_annotate, annotate_fontsize)

    ax1.set_title(title, font='Fira Sans', fontsize=28)
    ax1.set_xlabel(x_label, font='Fira Sans', fontsize=20)
    ax1.set_ylabel(y_label, font='Fira Sans', fontweight='bold', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=18)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(False)
    font_properties = get_font_properties('Fira Sans', 14)

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
        driver = qualy[qualy['position'] == best_pos]['driverCode'].min()
        qualy_result.append(driver)

    print(Counter(race_result))
    print(f'QUALYS: {Counter(qualy_result)}')
    print(f'{d_1} points: {d1_points}')
    print(f'{d_2} points: {d2_points}')


def get_pit_stops_ergast(year):
    """
        Get pitstops data from ergast

        Parameters:
        year (int): Year of analysis

   """
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
    """
        Get all circuits in F1 history


   """

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
    """
        Get top speeds for a range of rounds

        Parameters:
        gp (int): Number of rounds

   """

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


def get_fastest_data(session, column='Speed', fastest_lap=False, DRS=True):
    """
        Get the fastest data in a session

        Parameters:
        session (Session): Session to be analyzed
        column (str): Data to plot
        fastest_lap (bool): Get only data from the fastest lap

   """

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    drivers = session.laps['Driver'].groupby(session.laps['Driver']).size()

    drivers = drivers.reset_index(name='Count')['Driver'].to_list()
    drivers.remove('OCO')
    drivers.remove('PIA')
    circuit_speed = {}
    colors_dict = {}

    for driver in drivers:
        d_laps = session.laps.pick_driver(driver).pick_quicklaps()
        if fastest_lap:
            d_laps = session.laps.pick_driver(driver).pick_fastest()
        if len(d_laps) > 0:
            if column == 'Speed':
                if not DRS:
                    d_laps = d_laps.telemetry[d_laps.telemetry['DRS'] >= 10]
                top_speed = max(d_laps.telemetry['Speed'])
            else:
                top_speed = round(min(d_laps[column]).total_seconds(), 3)
            if fastest_lap:
                team = d_laps['Team'].lower()
            else:
                team = d_laps['Team'].values[0].lower()
            if team == 'red bull racing':
                team = 'red bull'
            elif team == 'haas f1 team':
                team = 'haas'
            circuit_speed[driver] = top_speed
            colors_dict[driver] = team

        print(circuit_speed)

    if column == 'Speed':
        order = True
        if fastest_lap:
            column = 'Top Speeds (only the fastest lap from each driver)'
        else:
            column = 'Top Speeds'
        x_fix = 5
        y_fix = 0.25
        annotate_fontsize = 11
    else:
        y_fix = 0.025
        x_fix = 0.45
        annotate_fontsize = 8
        order = False
        column = f"{column[:-5]} {column[-5:-4]} Times"
        if not DRS:
            column += 'without DRS'

    circuit_speed = {k: v for k, v in sorted(circuit_speed.items(), key=lambda item: item[1], reverse=order)}

    fig, ax1 = plt.subplots(figsize=(8, 6.5), dpi=175)

    colors = []
    for i in range(len(circuit_speed)):
        colors.append(fastf1.plotting.TEAM_COLORS[colors_dict[list(circuit_speed.keys())[i]]])

    bars = ax1.bar(list(circuit_speed.keys()), list(circuit_speed.values()), color=colors,
                   edgecolor='white')

    round_bars(bars, ax1, colors, y_offset_rounded=0.05)
    annotate_bars(bars, ax1, y_fix, annotate_fontsize, text_annotate='default', ceil_values=False)

    ax1.set_title(f'{column} in {str(session.event.year) + " " + session.event.Country + " " + session.name}',
                  font='Fira Sans', fontsize=12)
    ax1.set_xlabel('Driver', fontweight='bold', fontsize=12)
    if 'Speed' in column:
        y_label = 'Max speed'
    else:
        y_label = 'Sector time'
    ax1.set_ylabel(y_label, fontweight='bold', fontsize=12)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(False)


    max_value = max(circuit_speed.values())
    ax1.set_ylim(min(circuit_speed.values()) - x_fix, max_value + x_fix)

    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)

    def format_ticks(val, pos):
        return '{:.0f}'.format(val)

    if column == 'Speed':
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{column} IN {str(session.event.year) + " " + session.event.Country + " " + session.name}',
                dpi=450)
    plt.show()


def wins_and_poles_circuit(circuit, start=None, end=None):
    """
        Get all wins and poles in a circuit

        Parameters:
        circuit (str): Circuit to analyze
        start (int): Year of start
        end (int): Year of end

   """

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
    """
        Get all the overtakes since 1999

   """
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


def plot_circuit():
    """
        Plot a 3D cirucit

   """

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
        data = My_Ergast().get_qualy_results([2023])
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
        annotate_bars(bars, ax, 0.2, 10.5, text_annotate='default', ceil_values=False)

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
            session_data = gp[(gp['driverId'] == driver) & (gp['constructorId'] == team)]
            if len(session_data) > 0:
                position.append(session_data['position'].values[0])
            else:
                print(f'{driver} not in {team}')

        print(np.round(np.mean(position), 2))
        return np.round(np.mean(position), 2), statistics.median(position)


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
    """
        Get the fastest lap of a circuit

        Parameters:
        circuit (str): A specific driver
        start (int): Year of start
        end (int): Year of end
        all_drivers (bool): Top 10 or all drivers

   """

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