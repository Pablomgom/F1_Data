import re

import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from matplotlib.patches import Patch
from fastf1 import plotting

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from datetime import timedelta
import math

from matplotlib.ticker import FuncFormatter


def race_pace_teammates(team):

    circuits = []
    legend = []
    color = []
    differences = []
    context = ''
    for i in range(14):

        race = fastf1.get_session(2023, i + 1, 'R')
        race.load()
        drivers = np.unique(race.laps.pick_team(team)['Driver'].values)
        n_laps_d1 = len(race.laps.pick_driver(drivers[0]).pick_quicklaps())
        n_laps_d2 = len(race.laps.pick_driver(drivers[1]).pick_quicklaps())
        grid_pos_d1 = race.results[race.results['Abbreviation'] == drivers[0]]['GridPosition']
        grid_pos_d2 = race.results[race.results['Abbreviation'] == drivers[1]]['GridPosition']
        final_pos_d1 = race.results[race.results['Abbreviation'] == drivers[0]]['Position']
        final_pos_d2 = race.results[race.results['Abbreviation'] == drivers[1]]['Position']
        circuits.append(race.event.Location.split('-')[0])

        if n_laps_d2 >= 25 and n_laps_d1 >= 25:

            n_total_laps = min(n_laps_d2, n_laps_d1)

            team_1_laps = race.laps.pick_driver(drivers[0])
            team_2_laps = race.laps.pick_driver(drivers[1])

            '''
            sc_laps = team_1_laps.pick_track_status('4567', 'any')
            n_stints_t1 = team_1_laps[~team_1_laps['LapNumber'].isin(sc_laps['LapNumber'].values)]
            n_stints_t1 = n_stints_t1[~np.isnan(n_stints_t1['PitInTime'])]
            n_stints_t1 = n_stints_t1[]
            n_stints_t1 = n_stints_t1['LapNumber'].values - 2


            n_stints_t2 = team_2_laps[~team_2_laps['LapNumber'].isin(sc_laps['LapNumber'].values)]
            n_stints_t2 = n_stints_t2[~np.isnan(n_stints_t2['PitInTime'])]
            n_stints_t2 = n_stints_t2['LapNumber'].values - 2
            '''
            max_laps = min(len(team_1_laps), len(team_2_laps))

            team_1_laps = team_1_laps[:max_laps].pick_quicklaps().pick_wo_box()
            team_2_laps = team_2_laps[:max_laps].pick_quicklaps().pick_wo_box()
            '''
            seconds_box = 24
            stints_t1 = len([i for i in n_stints_t1 if i in team_1_laps['LapNumber'].values])
            stints_t2 = len([i for i in n_stints_t2 if i in team_1_laps['LapNumber'].values])
            '''
            sum_t1 = team_1_laps['LapTime'].sum()
            sum_t2 = team_2_laps['LapTime'].sum()
            '''
            sum_t1 += pd.Timedelta(seconds=((stints_t1 - 1) * seconds_box))
            sum_t2 += pd.Timedelta(seconds=((stints_t2 - 1) * seconds_box)))
            '''
            mean_t1 = sum_t1 / len(team_1_laps)
            mean_t2 = sum_t2 / len(team_2_laps)

            if mean_t1 > mean_t2:
                legend.append(f'{drivers[1]} faster')
                color.append('blue')
            else:
                legend.append(f'{drivers[0]} faster')
                if drivers[0] == 'RIC':
                    color.append('white')
                elif drivers[0] == 'LAW':
                    color.append('green')
                else:
                    color.append('orange')

            original_value = mean_t1.total_seconds()
            new_value = mean_t2.total_seconds()

            delta_diff = ((new_value - original_value) / original_value) * 100
            differences.append(round(delta_diff, 2))

            stint_d1 = race.laps.pick_driver(drivers[0]).pick_quicklaps().pick_wo_box()[:n_total_laps]['Stint'].value_counts()
            stint_d2 = race.laps.pick_driver(drivers[1]).pick_quicklaps().pick_wo_box()[:n_total_laps]['Stint'].value_counts()

            stint_d1 = stint_d1.sort_index()
            stint_d1.index = [1.0 if idx == stint_d1.index[0] else idx for idx in stint_d1.index]

            tyres_d1 = pd.Series(dtype='str')
            for i, stint in stint_d1.items():
                data = race.laps[race.laps['Stint'] == i]
                tyre = pd.Series([data.pick_driver(drivers[0])['Compound'].min()])
                tyre[0] = f'{str(int(i))} -> {tyre[0]}'
                tyres_d1 = tyres_d1._append(tyre)

            stint_d2 = stint_d2.sort_index()
            stint_d2.index = [1.0 if idx == stint_d2.index[0] else idx for idx in stint_d2.index]

            tyres_d2 = pd.Series(dtype='str')
            for i, stint in stint_d2.items():
                data = race.laps[race.laps['Stint'] == i]
                tyre = pd.Series([data.pick_driver(drivers[1])['Compound'].min()])
                tyre[0] = f'{str(int(i))} -> {tyre[0]}'
                tyres_d2 = tyres_d2._append(tyre)

            def get_strategy(stints, tyres, add_blank=None):

                formatted_stints = []
                for elem1, elem2 in zip(stints, tyres):
                    formatted_stints.append(f'{elem2} ({elem1})')

                final_string = ' - '.join(formatted_stints) + '\n'
                if add_blank:
                    final_string += '\n'
                return final_string

            final_string_d1 = get_strategy(stint_d1, tyres_d1)
            final_string_d2 = get_strategy(stint_d2, tyres_d2, add_blank=True)

            context += race.event.Location.split('-')[0].upper() + '\n'
            context += f'{drivers[0]} from P{int(grid_pos_d1.max())} to P{int(final_pos_d1.max())}: '
            context += final_string_d1
            context += f'{drivers[1]} from P{int(grid_pos_d2.max())} to P{int(final_pos_d2.max())}: '
            context += final_string_d2

        else:
            differences.append(0)
            legend.append(f'{drivers[1]} faster')
            color.append('blue')

    fig, ax = plt.subplots(figsize=(18, 10))

    # Create bar objects and keep them in a list
    bars = []

    for c, d, col in zip(circuits, differences, color):
        bar = ax.bar(c, d, color=col)
        bars.append(bar)

    for i in range(len(differences)):
        if differences[i] > 0:  # If the bar is above y=0
            plt.text(circuits[i], differences[i] + 0.03, str(differences[i]) + '%',
                     ha='center', va='top', fontsize=12)
        elif differences[i] < 0:  # If the bar is below y=0
            plt.text(circuits[i], differences[i] - 0.03, str(differences[i]) + '%',
                     ha='center', va='bottom', fontsize=12)

    plt.axhline(0, color='white', linewidth=0.8)

    # Convert your list to a Pandas Series
    differences_series = pd.Series(differences)

    # Calculate the rolling mean
    mean_y = differences_series.rolling(window=4, min_periods=1).mean().tolist()

    plt.plot(circuits, mean_y, color='red',
             marker='o', markersize=5.5, linewidth=3.5, label='Moving Average (4 last races)')

    # Create legend using unique labels
    legend_unique = list(set(legend))  # unique legend items
    legend_handles = {l: None for l in legend_unique}  # initialize dictionary to hold legend handles

    # Loop through to find the corresponding handles for unique legend items
    for l, b in zip(legend, bars):
        if legend_handles[l] is None:
            legend_handles[l] = b[0]

    # Create handles and labels list for the legend
    all_legend_handles = [legend_handles[l] for l in legend_unique]
    all_legend_labels = legend_unique

    # Add the handle and label for the Moving Average
    ma_handle = plt.Line2D([0], [0], marker='o', color='red', linestyle='-', linewidth=3.5, markersize=5.5)
    all_legend_handles.append(ma_handle)
    all_legend_labels.append('Moving Average (4 last races)')

    plt.legend(all_legend_handles, all_legend_labels, fontsize='x-large', loc='upper left')

    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')
    plt.title(f'RACE PACE COMPARATION BETWEEN {team.upper()} TEAMMATES', fontsize=26)
    plt.xlabel('Circuit', fontsize=16)
    plt.ylabel('Time diff (seconds)', fontsize=16)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'../PNGs/RACE DIFF BETWEEN {team} TEAMMATES.png', dpi=450)
    plt.show()
    print(context)


def position_changes(session):
    plotting.setup_mpl(misc_mpl_mods=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.9))

    for drv in session.drivers:
        drv_laps = session.laps.pick_driver(drv)
        abb = drv_laps['Driver'].iloc[0]

        if abb == 'MSC':
            color = '#cacaca'
        elif abb == 'VET':
            color = '#006f62'
        elif abb == 'LAT':
            color = '#012564'
        elif abb == 'LAW':
            color = '#2b4562'
        else:
            color = plotting.driver_color(abb)

        starting_grid = session.results.GridPosition.to_frame().reset_index(drop=False).rename(
            columns={'index': 'driver'})
        dict_drivers = session.results.Abbreviation.to_dict()

        starting_grid = starting_grid.replace({'driver': dict_drivers}).replace({'GridPosition': {0: 20}})

        vueltas = drv_laps['LapNumber'].reset_index(drop=True)
        position = drv_laps['Position'].reset_index(drop=True)
        position.at[0] = starting_grid[starting_grid['driver'] == abb]['GridPosition']

        ax.plot(vueltas, position,
                label=abb, color=color)

    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')

    ax.legend(bbox_to_anchor=(1.0, 1.02))

    plt.title(session.event.OfficialEventName, fontsize=14, loc='center')
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../PNGs/POSITION CHANGES {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def driver_lap_times(race, driver, fastest_laps=True):
    plotting.setup_mpl(misc_mpl_mods=False)

    driver_laps = race.laps.pick_driver(driver)
    max_stint = driver_laps['Stint'].value_counts().index[0]
    driver_laps_filter = driver_laps[driver_laps['Stint'] == max_stint]

    # Convert to timedelta series
    series = driver_laps_filter['LapTime']

    # Calculate Q1 and Q3
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    filtered_series = series[~((series < (Q1 - 1 * IQR)) | (series > Q3))]
    driver_laps_filter['LapTime'] = filtered_series
    driver_laps_filter = driver_laps_filter[driver_laps_filter['LapTime'].notna()]
    driver_laps_filter['LapNumber'] = driver_laps_filter['LapNumber'] - driver_laps_filter['LapNumber'].min() + 1
    driver_laps_filter['LapTime_seconds'] = driver_laps_filter['LapTime'].dt.total_seconds()
    driver_laps_filter['MovingAverage'] = driver_laps_filter['LapTime_seconds'].rolling(window=3, min_periods=1).mean()
    driver_laps_filter = driver_laps_filter.reset_index()

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(data=driver_laps_filter,
                    x="LapNumber",
                    y="LapTime_seconds",
                    ax=ax,
                    hue="Compound",
                    palette=plotting.COMPOUND_COLORS,
                    s=125,
                    linewidth=0,
                    legend='auto')

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")
    plt.grid(color='w', which='major', axis='both', linestyle='--')
    # The y-axis increases from bottom to top by default
    # Since we are plotting time, it makes sense to invert the axis
    text = 'Laptimes in Long Stints'

    def format_func(value, tick_number):
        minutes = np.floor(value / 60)
        seconds = int(value % 60)
        milliseconds = int((value * 1000) % 1000)  # multiplying by 1000 to get milliseconds
        return f"{int(minutes)}:{seconds:02}.{milliseconds:03}"

    ax.plot(driver_laps_filter['LapNumber'], driver_laps_filter['MovingAverage'], label="Avg of the last 3 laps",
            color=(0, 191 / 255, 255 / 255))

    # 1. Plot the MA points
    sns.scatterplot(data=driver_laps_filter,
                    x="LapNumber",
                    y="MovingAverage",
                    ax=ax,
                    color=(0, 191 / 255, 255 / 255),
                    s=125,
                    linewidth=0,
                    legend=False,
                    zorder=-5)

    # Annotate the moving average values:
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="none", lw=0)

    for lap, time in zip(driver_laps_filter['LapNumber'], driver_laps_filter['LapTime_seconds']):
        if not np.isnan(time):
            ax.annotate(f"{format_func(time, None)}",
                        (lap, time),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        color='red',
                        bbox=bbox_props)  # Adding the white box here

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.legend()
    # Turn on major grid lines
    sns.despine(left=True, bottom=True)


    plt.tight_layout()
    plt.savefig(f"../PNGs/{driver} LAPS {race.event.OfficialEventName}.png", dpi=400)
    plt.show()


def driver_race_times_per_tyre(race, driver):
    # The misc_mpl_mods option enables minor grid lines which clutter the plot
    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    driver_laps = race.laps.pick_driver(driver).pick_quicklaps().reset_index()

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(data=driver_laps,
                    x="LapNumber",
                    y="LapTime",
                    ax=ax,
                    hue="Compound",
                    palette=fastf1.plotting.COMPOUND_COLORS,
                    s=80,
                    linewidth=0,
                    legend='auto')

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")

    # The y-axis increases from bottom to top by default
    # Since we are plotting time, it makes sense to invert the axis
    ax.invert_yaxis()
    plt.suptitle(f"{driver} Laptimes in {race.event.OfficialEventName}")

    # Turn on major grid lines
    plt.grid(color='w', which='major', axis='both')
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(f"../PNGs/RACE LAPS {driver} {race.event.OfficialEventName}.png", dpi=400)
    plt.show()

def tyre_strategies(session):
    laps = session.laps
    drivers = session.drivers

    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

    stints = laps[["Driver", "Stint", "Compound", "LapNumber", "FreshTyre", "TyreLife"]]

    past_stint = 1
    past_driver = stints.loc[0, 'Driver']
    for i in range(len(stints)):
        changed = False
        current_driver = stints.loc[i, 'Driver']

        if past_driver != current_driver:
            past_stint = 1

        current_stint = stints.loc[i, 'Stint']
        if past_stint != current_stint:
            tyre_laps_prev = stints.loc[i - 1, 'TyreLife']
            tyre_laps = stints.loc[i, 'TyreLife']

            if tyre_laps_prev + 1 == tyre_laps:
                indexes = stints.index[stints['Driver'] == stints.loc[i, 'Driver']].tolist()
                indexes = [x for x in indexes if x >= i]
                for index in indexes:
                    stints.loc[index, 'Stint'] = stints.loc[index, 'Stint'] - 1
                    stints.loc[index, 'FreshTyre'] = stints.loc[min(indexes) - 1, 'FreshTyre']
                past_stint = current_stint - 1
                changed = True

        if not changed:
            past_stint = current_stint
            past_driver = current_driver

    stints = stints.groupby(["Driver", "Stint", "Compound", "FreshTyre"])
    stints = stints.count().reset_index()

    stints = stints.rename(columns={"LapNumber": "StintLength"})

    fig, ax = plt.subplots(figsize=(5, 10))

    patches = {}

    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]
        if driver == 'ALB':
            a = 1

        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            # each row contains the compound name and stint length
            # we can use these information to draw horizontal bars

            if row['FreshTyre']:
                alpha = 1
                color = plotting.COMPOUND_COLORS[row["Compound"]]
            else:
                alpha = 0.65
                rgb_color = mcolors.to_rgb(plotting.COMPOUND_COLORS[row["Compound"]])
                color = tuple([x * 0.8 for x in rgb_color])

            plt.barh(
                y=driver,
                width=row["StintLength"],
                left=previous_stint_end,
                color=color,
                edgecolor="black",
                fill=True,
                alpha=alpha
            )

            label = f"{'New' if row['FreshTyre'] else 'Used'} {row['Compound']}"
            color_rgb = mcolors.to_rgb(color)  # Convert color name to RGB
            patches[label] = mpatches.Patch(color=color_rgb + (alpha,), label=label)

            previous_stint_end += row["StintLength"]

    patches = {k: patches[k] for k in sorted(patches.keys(), key=lambda x: (x.split()[1], x.split()[0]))}
    plt.legend(handles=patches.values(), bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)

    fig.suptitle(session.event.OfficialEventName, fontsize=12)
    plt.xlabel("Lap Number")
    plt.grid(False)
    # invert the y-axis so drivers that finish higher are closer to the top
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    plt.savefig(f"../PNGs/TYRE STRATEGY {session.event.OfficialEventName}.png", dpi=400)

    plt.show()


def driver_laptimes(race):

    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

    point_finishers = race.drivers[:10]
    point_finishers = ['1', '11', '55', '16', '63', '44', '23', '4', '14', '77']
    driver_laps = race.laps.pick_drivers(point_finishers).pick_quicklaps()
    driver_laps = driver_laps.reset_index()
    finishing_order = [race.get_driver(i)["Abbreviation"] for i in point_finishers]

    driver_colors = {abv: fastf1.plotting.DRIVER_COLORS[driver] for
                     abv, driver in fastf1.plotting.DRIVER_TRANSLATE.items()}

    fig, ax = plt.subplots(figsize=(10, 5))
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

    sns.violinplot(data=driver_laps,
                   x="Driver",
                   y="LapTime(s)",
                   inner=None,
                   scale="area",
                   order=finishing_order,
                   palette=driver_colors,
                   )

    sns.swarmplot(data=driver_laps,
                  x="Driver",
                  y="LapTime(s)",
                  order=finishing_order,
                  hue="Compound",
                  palette=fastf1.plotting.COMPOUND_COLORS,
                  hue_order=["SOFT", "MEDIUM", "HARD"],
                  linewidth=0,
                  size=5,
                  )

    plt.ylim(min(driver_laps['LapTime']).total_seconds() - 2,
             max(driver_laps['LapTime']).total_seconds() + 2)  # change these numbers as per your needs
    ax.set_xlabel("Driver")
    ax.set_ylabel("Lap Time (s)")
    plt.suptitle(f"{race.event['EventDate'].year} {race.event['EventName']} Lap Time Distributions")
    sns.despine(left=False, bottom=False)

    plt.savefig(f"../PNGs/{race.event['EventDate'].year} {race.event['EventName']} Lap Time Distributions.png", dpi=400)
    plt.tight_layout()
    plt.show()


def race_diff(team_1, team_2, year):

    races = []
    session_names = []
    context = ''
    team_1_times = []
    team_2_times = []

    team_1_total_laps = []
    team_2_total_laps = []
    n_races = Ergast().get_race_results(season=year, limit=1000)
    for i in range(len(n_races.content)):
        race = fastf1.get_session(year,  i + 1, 'R')
        race.load(telemetry=True)
        races.append(race)
        session_names.append(race.event['Location'].split('-')[0])

        min_pos_t1 = race.results[race.results['TeamName'] == team_1]['Position'].min()
        d_t1 = race.results[race.results['Position'] == min_pos_t1]['Abbreviation'].min()

        min_pos_t2 = race.results[race.results['TeamName'] == team_2]['Position'].min()
        d_t2 = race.results[race.results['Position'] == min_pos_t2]['Abbreviation'].min()

        team_1_laps = race.laps.pick_driver(d_t1)
        team_2_laps = race.laps.pick_driver(d_t2)

        max_laps = min(len(team_1_laps), len(team_2_laps))

        team_1_laps = team_1_laps[:max_laps].pick_quicklaps().pick_wo_box()
        team_2_laps = team_2_laps[:max_laps].pick_quicklaps().pick_wo_box()

        sum_t1 = team_1_laps['LapTime'].sum()
        sum_t2 = team_2_laps['LapTime'].sum()

        mean_t1 = sum_t1 / len(team_1_laps)
        if len(team_2_laps) == 0:
            team_2_times.append(0)
            team_2_total_laps.append(len(team_2_laps))
        else:
            mean_t2 = sum_t2 / len(team_2_laps)
            team_2_times.append(mean_t2)
            team_2_total_laps.append(len(team_2_laps))

        team_1_times.append(mean_t1)
        team_1_total_laps.append(len(team_1_laps))

    delta_laps = []

    for i in range(len(team_2_times)):
        mean_time_team_1 = team_1_times[i]
        mean_time_team_2 = team_2_times[i]

        if team_1_total_laps[i] > team_2_total_laps[i]:
            laps = team_1_total_laps[i]
        else:
            laps = team_2_total_laps[i]

        if mean_time_team_2 == 0:
            delta_laps.append(0)
        else:
            delta = ((mean_time_team_2 - mean_time_team_1) / mean_time_team_2) * laps
            delta_laps.append(delta)

    fig, ax1 = plt.subplots(figsize=(25, 12))
    delta_laps = [x if not math.isnan(x) else 0 for x in delta_laps]

    for i in range(len(session_names)):
        color = plotting.team_color(team_1) if delta_laps[i] > 0 else plotting.team_color(team_2)
        label = f'{team_1} faster' if delta_laps[i] > 0 else f'{team_2} faster'
        plt.bar(session_names[i], delta_laps[i], color=color, label=label)

    # Add exact numbers above or below every bar based on whether it's a maximum or minimum
    for i in range(len(session_names)):
        if delta_laps[i] > 0:  # If the bar is above y=0
            plt.text(session_names[i], delta_laps[i] + 0.04, "{:.2f} %".format(delta_laps[i]),
                     ha='center', va='top')
        else:  # If the bar is below y=0
            plt.text(session_names[i], delta_laps[i] - 0.04, "{:.2f} %".format(delta_laps[i]),
                     ha='center', va='bottom')

    # Set the labels and title
    plt.ylabel(f'Percentage time difference', fontsize=16)
    plt.xlabel('Circuit', fontsize=16)
    ax1.yaxis.grid(True, linestyle='--')
    plt.title(f'{team_1} VS {team_2} {year} race time difference', fontsize=24)

    step = 0.2

    if min(delta_laps) < 0:
        start = np.floor(min(delta_laps) / step) * step
    else:
        start = np.ceil(min(delta_laps) / step) * step
    end = np.ceil(max(delta_laps) / step) * step

    delta_laps = pd.Series(delta_laps)
    mean_y = list(delta_laps.rolling(window=4, min_periods=1).mean())

    plt.plot(session_names, mean_y, color='red',
             marker='o', markersize=4, linewidth=2, label='Moving Average (4 last races)')

    if min(delta_laps) < 0:
        plt.axhline(0, color='white', linewidth=2)

    # Generate a list of ticks from minimum to maximum y values considering 0.0 value and step=0.2
    yticks = list(np.arange(start, end + step, step))
    yticks = sorted(yticks)

    plt.yticks(yticks, [f'{tick:.2f} %' for tick in yticks])

    # To avoid repeating labels in the legend, we handle them separately
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.savefig(f"../PNGs/{team_2} VS {team_1} {year} race time difference.png", dpi=400)

    # Show the plot
    plt.show()

    print(context)


def race_distance(session, driver_1, driver_2):
    laps_driver_1 = session.laps.pick_driver(driver_1).reset_index()
    laps_driver_2 = session.laps.pick_driver(driver_2).reset_index()

    laps_driver_1_box = pd.isna(laps_driver_1['PitInTime'])
    filtered_laps_driver_1 = laps_driver_1[~laps_driver_1_box]
    pit_laps_driver_1 = np.array(filtered_laps_driver_1.index + 1)

    laps_driver_2_box = pd.isna(laps_driver_2['PitInTime'])
    filtered_laps_driver_2 = laps_driver_2[~laps_driver_2_box]
    pit_laps_driver_2 = np.array(filtered_laps_driver_2.index + 1)

    laps_diff = []
    laps = []
    for i in range(len(laps_driver_1)):
        laps_diff.append(laps_driver_1['Time'][i].total_seconds() - laps_driver_2['Time'][i].total_seconds())
        laps.append(i+1)

    laps_diff = [0 if math.isnan(x) else x for x in laps_diff]
    progressive_sum = laps_diff
    colors = ['red']
    for i in range(len(progressive_sum) - 1):
        if progressive_sum[i] < progressive_sum[i + 1]:
            colors.append('green')
        else:
            colors.append('red')
    plt.figure(figsize=(37, 8))
    # Bar Plot
    bars = plt.bar(laps, progressive_sum, color=colors, width=0.9)

    # Annotate bars with their values
    for bar in bars:
        yval = bar.get_height()
        offset = 0.7 if yval > 0 else -0.7  # This will adjust the position above or below the bar. Modify the value if needed.
        plt.annotate(
            f'{yval:.2f}',  # Format to 2 decimal places, modify as needed
            (bar.get_x() + bar.get_width() / 2, yval + offset),  # Adjusted the y-coordinate here
            ha='center',  # horizontal alignment
            va='center',  # vertical alignment
            fontsize=8 # font size
        )
    # Create custom legend

    legend_elements = [Patch(facecolor='green', edgecolor='green', label=f'{driver_2} Faster'),
                       Patch(facecolor='red', edgecolor='red', label=f'{driver_1} Faster')]
    plt.legend(handles=legend_elements, loc='best', fontsize=20)

    # Given offset for overlapping text
    text_offset = 3

    for pit_lap in pit_laps_driver_1:
        end_y = progressive_sum[pit_lap - 1]

        # If end_y is positive or zero, adjust the start_y and dy to ensure arrow's head ends at y=0
        if end_y >= 0:
            start_y = -15
            dy = 13
            text_y = -18
        else:
            start_y = end_y - 15
            dy = 10
            text_y = start_y - 1.5

        # Check for overlap and adjust
        if pit_lap in pit_laps_driver_2:
            text_y -= text_offset

        plt.arrow(pit_lap, start_y, 0, dy, head_width=0.15, head_length=2, color='white')
        plt.text(pit_lap, text_y, f'{driver_1} Pit Stop', color='white', fontsize=11, ha='center')

    for pit_lap in pit_laps_driver_2:
        end_y = progressive_sum[pit_lap - 1]

        # If end_y is positive or zero, adjust the start_y and dy to ensure arrow's head ends at y=0
        if end_y >= 0:
            start_y = -15
            dy = 13
            text_y = -16.25
        else:
            start_y = end_y - 15
            dy = 10
            text_y = start_y - 1.25

        if pit_lap + 1 in pit_laps_driver_1 and pit_lap + 1 not in pit_laps_driver_2:
            text_y -= 2

        plt.arrow(pit_lap, start_y, 0, dy, head_width=0.15, head_length=2, color='white')
        plt.text(pit_lap, text_y, f'{driver_2} Pit Stop', color='white', fontsize=11, ha='center')

    plt.xlabel('Laps', fontsize=20)
    plt.ylabel('Progressive Time Difference (seconds)', fontsize=20)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.title(f'Progressive Time Difference between {driver_1} and {driver_2} in {session.event["EventName"] + " " + str(session.event["EventDate"].year)}', fontsize=20)
    plt.grid(True, axis='y')


    # Display the plot
    plt.tight_layout()  # Adjusts plot parameters for a better layout
    plt.savefig(f'../PNGs/Progressive Time Difference between {driver_1} and {driver_2} in {session.event["EventName"] + " " + str(session.event["EventDate"].year)}', dpi=400)
    plt.show()




