import fastf1
import numpy as np
import pandas as pd
import re

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

from src.general_analysis.race_plots import rounded_top_rect
from src.general_analysis.table import render_mpl_table

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from src.variables.variables import team_colors_2023


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


def mean_points_per_team(year):

    ergast = Ergast()
    races = ergast.get_race_results(season=year, limit=1000)
    teams = set(races.content[0]['constructorName'])
    team_points = pd.DataFrame(columns=['Team', 'Points', 'Circuit'])
    circuits = np.array(races.description['circuitId'])
    circuits = [i.replace('_', ' ').title() for i in circuits]
    for i in range(len(races.content)):
        for team in teams:
            points = races.content[i][races.content[i]['constructorName'] == team]['points'].sum()
            row = [team, points, circuits[i]]
            team_points = team_points._append(pd.Series(row, index=team_points.columns), ignore_index=True)

    team_categories = pd.Categorical(team_points['Team'], categories=team_points['Team'].unique(), ordered=True)
    race_categories = pd.Categorical(team_points['Circuit'], categories=team_points['Circuit'].unique(), ordered=True)
    ct = pd.crosstab(team_categories, race_categories, values=team_points['Points'], aggfunc='sum')
    ma_points = ct.rolling(window=4, min_periods=1, axis=1).mean()
    ordered_colors = [team_colors_2023[team] for team in ma_points.index]
    transposed = ma_points.transpose()
    ax = transposed.plot(figsize=(12, 12), marker='o', color=ordered_colors)

    font = FontProperties(family='Fira Sans', size=12)
    plt.title(f"Average points in the last 4 races", font='Fira Sans', fontsize=28)
    plt.xlabel("Races", font='Fira Sans', fontsize=18)
    plt.ylabel("Avg. Points", font='Fira Sans', fontsize=18)
    plt.legend(prop=font, loc="upper left", bbox_to_anchor=(1.0, 0.6))
    plt.xticks(ticks=range(len(transposed)), labels=transposed.index,
               rotation=90, fontsize=12, fontname='Fira Sans')
    yticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    plt.yticks(yticks, fontsize=12, fontname='Fira Sans')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjusts the plot layout for better visibility
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.savefig(f'../PNGs/AVERAGE POINTS.png', dpi=400)
    plt.show()


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

def cluster_circuits(year, rounds, prev_year, circuit, clusters=None):

    session_type = ['FP1', 'FP2', 'FP3', 'Q', 'S', 'SS', 'R']
    circuits = []

    data = []
    for i in range(0, rounds + 1):
        fast_laps = []
        for type in session_type:
            try:
                print(f'{i}: {type}')
                if i == rounds:
                    session = fastf1.get_session(prev_year, circuit, type)
                else:
                    session = fastf1.get_session(year, i + 1, type)
                session.load()
                fast_laps.append(session.laps.pick_fastest())
            except:
                print(f'{type} not in this event')

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
        corners_per_meter = length/corners
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
        brakes = telemetry[telemetry['Throttle'] == 100]
        full_brakes = len(brakes) / len(telemetry)
        data.append([corners_per_meter, max_speed, q1, median, q3, avg, full_gas])

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
                type = 'Low speed tracks'
            case 1:
                type = 'Medium speed tracks'
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

    legend_lines =[Line2D([0], [0], color='red', lw=4),
                   Line2D([0], [0], color='blue', lw=4),
                   Line2D([0], [0], color='green', lw=4)]

    plt.legend(legend_lines, ['Low speed tracks', 'Medium speed tracks', 'High speed tracks'],
               loc='upper right', fontsize='x-large')

    ax.axis('off')
    ax.grid(False)
    plt.title('SIMILARITY BETWEEN CIRCUITS', font='Fira Sans', fontsize=28)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig('../PNGs/Track clusters.png', dpi=400)
    plt.show()



def pitstops(year, round=None, exclude=None):
    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    pitstops = pd.read_csv('../resources/Pit stops.csv', sep='|')
    pitstops = pitstops[pitstops['Year'] == year]
    if round is None:
        pitstops = pitstops.groupby('Driver')['Time'].mean()
    else:
        pitstops = pitstops[pitstops['Race_ID'] == round]
    pitstops = pitstops.reset_index()
    pitstops = pitstops.sort_values(by='Time', ascending=True)
    pitstops['Time'] = pitstops['Time'].round(2)

    fig, ax1 = plt.subplots(figsize=(29, 10))
    drivers = [i for i in pitstops['Driver']]
    colors = []

    for driver in drivers:
        for key, value in fastf1.plotting.DRIVER_COLORS.items():
            parts = key.split(" ", 1)
            new_key = parts[1] if len(parts) > 1 else key
            if (new_key == driver.lower()) or (new_key == 'guanyu' and driver == 'Zhou'):
                colors.append(value)
                break

    if round is not None:
        name_count = {}

        def update_name(name):
            if name in name_count:
                name_count[name] += 1
            else:
                name_count[name] = 1
            return f"{name}_{name_count[name]}"

        pitstops['Driver'] = pitstops['Driver'].apply(update_name)

    if exclude is not None:
        pitstops = pitstops[~pitstops['Driver'].isin(exclude)]

    bars = ax1.bar(pitstops['Driver'], pitstops['Time'], color=colors,
                   edgecolor='white')

    for bar in bars:
        bar.set_visible(False)

    # Overlay rounded rectangle patches on top of the original bars
    i = 0
    for bar in bars:
        height = bar.get_height()
        x, y = bar.get_xy()
        width = bar.get_width()

        # Create a fancy bbox with rounded corners and add it to the axes
        rounded_box = rounded_top_rect(x, y, width, height, 0.1, colors[i])
        rounded_box.set_facecolor(colors[i])
        ax1.add_patch(rounded_box)
        i += 1


    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f'{height}', ha='center', va='bottom',
                 font='Fira Sans', fontsize=14)

    ax1.set_title(f'PIT STOP TIMES IN 2023', font='Fira Sans', fontsize=28)
    ax1.set_xlabel('Driver', font='Fira Sans', fontsize=20)
    ax1.set_ylabel('Time (s)', font='Fira Sans', fontweight='bold', fontsize=20)
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
    N = 12
    top_N = positions.nlargest(N)
    top_N['Other'] = positions.iloc[N:].sum()

    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)

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
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, colors=colors)  # Set line color to black

    ax.legend(title="Finish legend", loc="center left", labels=top_N.index, bbox_to_anchor=(0.8, 0.1))

    plt.title(f'{driver} finish history (Total races: {positions.sum()})', fontsize=16, color='white')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver} finish history', dpi=400)
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


def get_pit_stops(year):
    ergast = Ergast()
    schedule = ergast.get_race_schedule(season=year, limit=1000)
    circuits = schedule.circuitId.values
    circuits = [item.replace("_", " ").title() for item in circuits]

    n_pit_stops = []
    for i in range(len(schedule)):
        pit_stop = ergast.get_pit_stops(season=year, round=i + 1, limit=1000)
        n_pit_stops.append(len(pit_stop.content[0]))

    fig, ax1 = plt.subplots(figsize=(32, 8))

    total = sum(n_pit_stops)
    mean = round(total / len(n_pit_stops), 2)

    bars = ax1.bar(circuits, n_pit_stops, color="#AED6F1", edgecolor='white', label='Total Pitstops')
    ax1.set_title(f'PIT STOPS IN {year} (Total: {total} - Avg: {mean})')
    ax1.set_xlabel('Circuit', fontweight='bold')
    ax1.set_ylabel('Total Pitstops', fontweight='bold')
    ax1.yaxis.grid(False)
    ax1.xaxis.grid(False)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height}', ha='center', va='bottom', fontsize=14)

    ax2 = ax1.twinx()
    mean_pits = [round(i / 20, 2) for i in n_pit_stops]

    total = sum(mean_pits)
    mean = round(total / len(mean_pits), 2)

    ax2.plot(circuits, mean_pits, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=8,
             label='Avg PitStops per driver')
    ax2.set_ylabel(f'PitStops per driver (Avg per race: {mean})', fontweight='bold')
    ax2.yaxis.grid(True, linestyle='dashed')

    for i, value in enumerate(mean_pits):
        ax2.annotate(f'{value}', (i, value), textcoords="offset points", xytext=(0, -18), ha='center',
                     fontsize=12, bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='black'))

    # Add a legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

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


def get_topspeed():
    top_speed_array = []

    for i in range(12):

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


def get_topspeed_in_session(session, column='Speed', fastest_lap=None, DRS=True):
    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    circuit_speed = {}
    colors_dict = {}

    if fastest_lap is not None:
        drivers = session.laps['Driver'].groupby(session.laps['Driver']).size()

        drivers = list(drivers.reset_index(name='Count')['Driver'].values)

        for driver in drivers:
            lap = session.laps.pick_driver(driver).pick_fastest()
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
    for bar in bars:
        bar.set_visible(False)
    i = 0
    for bar in bars:
        height = bar.get_height()
        x, y = bar.get_xy()
        width = bar.get_width()
        # Create a fancy bbox with rounded corners and add it to the axes
        rounded_box = rounded_top_rect(x, y, width, height, 0.1, colors[i])
        rounded_box.set_facecolor(colors[i])
        ax1.add_patch(rounded_box)
        i += 1

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

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + y_fix, f'{height}', ha='center', va='bottom', fontsize=14)

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


def wins_in_circuit(circuit, start=None, end=None):
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


def day_all_races():
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


def overtakes():
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


def races_by_number(number):
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
        if driver == 'Fernando Alonso':
            a = 1
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
                                        if re.search(r'(Spun off|Accident|Collision|Finished|\+)', status_d1):
                                            if not re.search(r'(Spun off|Accident|Collision|Finished|\+)', status_d2):
                                                luck[driver] += 1
                                        else:
                                            if re.search(r'(Spun off|Accident|Collision|Finished|\+)', status_d2):
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
                        position = 'P'+str(position)
                    team = results['constructorName'].values[0]
                    print(f'{year}: From P{grid} to {position} with {team}')

