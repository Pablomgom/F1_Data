import fastf1
import numpy as np
import pandas as pd
import re
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt, cm
from collections import Counter
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.general_analysis.table import render_mpl_table


def pitstops(year, round=None):
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

    bars = ax1.bar(pitstops['Driver'], pitstops['Time'], color=colors,
                   edgecolor='white')

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f'{height}', ha='center', va='bottom', fontsize=14)

    ax1.set_title(f'PIT STOP TIMES IN 2023 ITALIAN GP', fontsize=28)
    ax1.set_xlabel('Driver', fontweight='bold', fontsize=20)
    ax1.set_ylabel('Avg time (s)', fontweight='bold', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=18)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(False)
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
                    positions = pd.concat([positions, pd.Series(['P'+str(race['position'].max())])], ignore_index=True)
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


def compare_drivers_season(d_1, d_2, season):

    ergast = Ergast()

    schedule = ergast.get_race_schedule(season=season, limit=1000)
    races = ergast.get_race_results(season=season, limit=1000)
    sprints = ergast.get_sprint_results(season=season, limit=1000)
    qualys = ergast.get_qualifying_results(season=season, limit=1000)
    total_races = races.content + sprints.content

    race_result = []
    qualy_result = []

    for race in total_races:
        best_pos = race[race['familyName'].isin([d_1, d_2])]['position'].min()
        driver = race[race['position'] == best_pos]['driverCode'].min()
        if race.shape[1] == 26:
            race_result.append(driver + ' - Race')
        elif race.shape[1] == 23:
            race_result.append(driver + ' - Sprint')

    for qualy in qualys.content:
        best_pos = qualy[qualy['familyName'].isin([d_1, d_2])]['position'].min()
        driver = qualy[qualy['position'] == best_pos]['driverCode'].min()
        qualy_result.append(driver)

    print(Counter(race_result))
    print(Counter(qualy_result))



def get_pit_stops(year):

    ergast = Ergast()
    schedule = ergast.get_race_schedule(season=year, limit=1000)
    circuits = schedule.circuitId.values
    circuits = [item.replace("_", " ").title() for item in circuits]

    n_pit_stops = []
    for i in range(len(schedule)):
        pit_stop = ergast.get_pit_stops(season=year, round=i+1, limit=1000)
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
    mean_pits = [round(i/20, 2) for i in n_pit_stops]

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

        session = fastf1.get_session(2023, i+1, 'Q')
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


def get_topspeed_in_session(session, column='Speed', DRS=None):

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    circuit_speed = {}
    colors_dict = {}

    if DRS is not None:
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
                    if lap[1].telemetry['DRS'].max() >= 10:
                        top_speed = 0
                    else:
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
        if DRS is not None:
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

    circuit_speed = {k: v for k, v in sorted(circuit_speed.items(), key=lambda item: item[1], reverse=order)}

    fig, ax1 = plt.subplots(figsize=(20, 8))

    colors = []
    for i in range(len(circuit_speed)):
        colors.append(fastf1.plotting.TEAM_COLORS[colors_dict[list(circuit_speed.keys())[i]]])

    bars = ax1.bar(list(circuit_speed.keys()), list(circuit_speed.values()), color=colors,
                   edgecolor='white')
    ax1.set_title(f'{column} in {str(session.event.year) + " " + session.event.Country + " " + session.name}')
    ax1.set_xlabel('Driver', fontweight='bold', fontsize=12)
    if column == 'Speed':
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
    plt.savefig(f'../PNGs/{column} IN {str(session.event.year) + " " + session.event.Country + " " + session.name}', dpi=400)
    plt.show()


def wins_in_circuit(circuit):

    ergast = Ergast()
    winners = pd.Series(dtype=str)
    poles = pd.Series(dtype=str)
    years = []
    for i in range(1950, 2024):
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

    # Create 2D plot
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

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
