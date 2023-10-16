import re
import statistics

from matplotlib.font_manager import FontProperties

from src.plots.plots import round_bars, annotate_bars
from src.exceptions import race_same_team_exceptions, qualy_exceptions, race_diff_team_exceptions
import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from fastf1 import plotting

from matplotlib import pyplot as plt, patches, image as mpimg
import seaborn as sns
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import math

def qualy_diff_teammates(team, rounds):
    """
         Prints the qualy diff between teammates

         Parameters:
         team (str): Team
         rounds (int): Rounds to be analyzed

    """
    from src.utils.utils import call_function_from_module
    circuits = []
    legend = []
    color = []
    differences = []

    for i in range(rounds):

        qualy = fastf1.get_session(2023, i + 1, 'Q')
        qualy.load()
        circuits.append(qualy.event.Location.split('-')[0])
        drivers = list(np.unique(qualy.laps.pick_team(team)['Driver'].values))
        q1, q2, q3 = qualy.laps.split_qualifying_sessions()
        if drivers[0] in q3['Driver'].unique() and drivers[1] in q3['Driver'].unique():
            session = q3
        elif drivers[0] in q2['Driver'].unique() and drivers[1] in q2['Driver'].unique():
            session = q2
        else:
            session = q1
        try:
            call_function_from_module(qualy_exceptions, f"{team.replace(' ', '_')}_{2023}", i + 1)
            d0_time = session.pick_driver(drivers[0]).pick_fastest()['LapTime'].total_seconds()
            d1_time = session.pick_driver(drivers[1]).pick_fastest()['LapTime'].total_seconds()

            if d0_time > d1_time:
                legend.append(f'{drivers[1]} faster')
                color.append('#0000FF')
            else:
                legend.append(f'{drivers[0]} faster')
                color.append('#FFA500')

            delta_diff = ((d0_time - d1_time) / d1_time) * 100
            differences.append(round(-delta_diff, 2))
        except AttributeError:
            print('No hay vuelta disponible')
            differences.append(np.nan)
            color.append('#0000FF')

    print(f'MEAN: {statistics.mean([i for i in differences if not np.isnan(i)])}')
    print(f'MEDIAN: {statistics.median([i for i in differences if not np.isnan(i)])}')

    fig, ax1 = plt.subplots(figsize=(7.2, 6.5), dpi=150)
    bars = plt.bar(circuits, differences, color=color)

    round_bars(bars, ax1, color)
    annotate_bars(bars, ax1, 0.01, 8, text_annotate='{height}%', ceil_values=False)

    legend_lines = []
    unique_colors = []
    unique_drivers = []
    i = 0
    for c_color in color:
        if c_color not in unique_colors:
            unique_colors.append(c_color)
            unique_drivers.append(legend[i])
            legend_p = Line2D([0], [0], color=c_color, lw=4)
            legend_lines.append(legend_p)
        i += 1

    plt.legend(legend_lines, unique_drivers,
               loc='lower left', fontsize='large')

    plt.axhline(0, color='white', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')
    plt.title(f'QUALY DIFFERENCE COMPARISON BETWEEN {team.upper()} TEAMMATES', font='Fira Sans', fontsize=2)
    plt.xticks(ticks=range(len(circuits)), labels=circuits,
               rotation=90, fontsize=12, fontname='Fira Sans')
    plt.xlabel('Circuit', font='Fira Sans', fontsize=16)
    plt.ylabel('Time diff (percentage)', font='Fira Sans', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'../PNGs/PACE DIFF BETWEEN {team} TEAMMATES.png', dpi=500)
    plt.show()


def apply_tyre_age_factor(d1, d2, mean_t1, mean_t2, avg_life_t1, avg_life_t2, team, country, location):

    """
         Applies to the race_pace a factor based on the tyres age

    """



    if team == 'Ferrari' and country == 'Spain':
        avg_life_t1 = 11
        avg_life_t2 = 11
    print(d1, avg_life_t1)
    print(d2, avg_life_t2)
    if country == 'Azerbaijan' or location == 'Jeddah' or location == 'Monaco':
        tyre_factor = 0.01
    elif location == 'Miami':
        tyre_factor = 0.02
    else:
        tyre_factor = 0.05
    tyre_diff = abs(avg_life_t1 - avg_life_t2)
    if avg_life_t1 > avg_life_t2:
        mean_t2 += (tyre_diff * tyre_factor)
    else:
        mean_t1 += (tyre_diff * tyre_factor)

    return mean_t1, mean_t2


def get_driver_race_pace(race, d1, d2, team, round, exceptions=True, return_og=False, team_mode=False, laps_mode='min'):
    """
         Get the race pace of a driver

    """
    team_1_laps = race.laps.pick_driver(d1)
    team_2_laps = race.laps.pick_driver(d2)
    min_laps = 0
    laps_d1 = len(team_1_laps)
    laps_d2 = len(team_2_laps)
    max_laps = min(laps_d1, laps_d2)
    if laps_mode != 'min':
        if laps_d1 < 25 or laps_d2 < 25:
            max_laps = max(laps_d1, laps_d2)

    if exceptions:
        from src.utils.utils import call_function_from_module
        min_laps, max_laps = call_function_from_module(race_same_team_exceptions,
                                                       f"{team.replace(' ', '_')}_{2023}",
                                                       round, max_laps)

    team_1_laps = team_1_laps[min_laps:max_laps].pick_quicklaps().pick_wo_box()
    team_2_laps = team_2_laps[min_laps:max_laps].pick_quicklaps().pick_wo_box()

    mean_t1 = team_1_laps['LapTime'].mean().total_seconds()
    mean_t2 = team_2_laps['LapTime'].mean().total_seconds()

    if team_mode and (laps_d1 >= 25 or laps_d2 >= 25):
        if laps_d1 < 25:
            mean_t1 = 1000000
        if laps_d2 < 25:
            mean_t2 = 1000000

    og_t1 = mean_t1
    og_t2 = mean_t2

    avg_life_tyre_t1 = team_1_laps['TyreLife'].mean()
    avg_life_tyre_t2 = team_2_laps['TyreLife'].mean()

    mean_t1, mean_t2 = apply_tyre_age_factor(d1, d2, mean_t1, mean_t2, avg_life_tyre_t1,
                                             avg_life_tyre_t2, team, race.event.Country, race.event.Location)

    if return_og:
        return og_t1, og_t2
    return mean_t1, mean_t2


def race_pace_teammates(team, rounds):
    """
         Plots the race diff between teammates

         Parameters:
         team (str): Team
         rounds (int): Rounds to be analyzed

    """


    circuits = []
    legend = []
    color = []
    differences = []
    for i in range(rounds):
        race = fastf1.get_session(2023, i + 1, 'R')
        race.load()
        drivers = list(np.unique(race.laps.pick_team(team)['Driver'].values))
        if team == 'Aston Martin' and 'STR' not in drivers:
            drivers.append('STR')
        elif team == 'Ferrari' and 'SAI' not in drivers:
            drivers.append('SAI')
        n_laps_d1 = len(race.laps.pick_driver(drivers[0]).pick_quicklaps().pick_wo_box())
        n_laps_d2 = len(race.laps.pick_driver(drivers[1]).pick_quicklaps().pick_wo_box())
        circuits.append(race.event.Location.split('-')[0])

        if n_laps_d2 >= 25 and n_laps_d1 >= 25:

            mean_t1, mean_t2 = get_driver_race_pace(race, drivers[0], drivers[1], team, i + 1)

            if mean_t1 > mean_t2:
                legend.append(f'{drivers[1]} faster')
                color.append('#0000FF')
            else:
                legend.append(f'{drivers[0]} faster')
                if drivers[0] == 'RIC':
                    color.append('#FFFFFF')
                elif drivers[0] == 'LAW':
                    color.append('#008000')
                else:
                    color.append('#FFA500')

            delta_diff = ((mean_t2 - mean_t1) / mean_t1) * 100
            if np.isnan(delta_diff):
                delta_diff = 0
            differences.append(round(delta_diff, 2))
        else:
            differences.append(0)
            legend.append(f'{drivers[1]} faster')
            color.append('#0000FF')

    mean_diff = [i for i in differences if i != 0]
    print(f'MEAN DIFF: {np.mean(mean_diff)}')
    print(f'MEDIAN DIFF: {statistics.median(mean_diff)}')

    fig, ax1 = plt.subplots(figsize=(7.2, 6.5), dpi=150)

    bars = plt.bar(circuits, differences, color=color)

    round_bars(bars, ax1, color, y_offset_rounded=0)
    annotate_bars(bars, ax1, 0.01, 8, text_annotate='{height}%', ceil_values=False)

    legend_lines = []
    unique_colors = []
    unique_drivers = []
    i = 0
    for c_color in color:
        if c_color not in unique_colors:
            unique_colors.append(c_color)
            unique_drivers.append(legend[i])
            legend_p = Line2D([0], [0], color=c_color, lw=4)
            legend_lines.append(legend_p)
        i += 1

    plt.legend(legend_lines, unique_drivers,
               loc='upper left', fontsize='large')

    plt.axhline(0, color='white', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')
    plt.title(f'RACE PACE COMPARISON BETWEEN {team.upper()} TEAMMATES', font='Fira Sans', fontsize=15)
    plt.xticks(ticks=range(len(circuits)), labels=circuits,
               rotation=90, fontsize=11, fontname='Fira Sans')
    plt.yticks(fontsize=11)
    plt.xlabel('Circuit', font='Fira Sans', fontsize=14)
    plt.ylabel('Time diff (percentage)', font='Fira Sans', fontsize=12)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', font='Fira Sans', fontsize=12, color='white', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'../PNGs/RACE DIFF BETWEEN {team} TEAMMATES.png', dpi=150)
    plt.show()


def position_changes(session):
    """
         Plots the position changes in every lap

         Parameters:
         session (Session): Session to analyze

    """

    plotting.setup_mpl(misc_mpl_mods=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.12))
    for drv in session.drivers:
        if drv != '55':
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
    ax.set_xlim([1, session.total_laps])
    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')
    ax.legend(bbox_to_anchor=(1.0, 1.02))

    plt.title(session.event.OfficialEventName + ' ' + session.name.upper(), font='Fira Sans', fontsize=18, loc='center')
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../PNGs/POSITION CHANGES {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def long_runs_FP2(race, driver):
    """
         Plots the long runs in FP2 for a given driver

         Parameters:
         race (Session): Session to analyze
         driver (str): Driver

    """

    plotting.setup_mpl(misc_mpl_mods=False)

    driver_laps = race.laps.pick_driver(driver)
    max_stint = driver_laps['Stint'].value_counts().index[0]
    driver_laps_filter = driver_laps[driver_laps['Stint'] == max_stint]

    # Convert to timedelta series
    series = driver_laps_filter['LapTime']

    # Calculate Q1 and Q3
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.55)
    IQR = Q3 - Q1

    filtered_series = series[~((series < (Q1 - 1 * IQR)) | (series > Q3))]
    driver_laps_filter['LapTime'] = filtered_series
    driver_laps_filter = driver_laps_filter[driver_laps_filter['LapTime'].notna()]
    driver_laps_filter['LapNumber'] = driver_laps_filter['LapNumber'] - driver_laps_filter['LapNumber'].min() + 1
    driver_laps_filter['LapTime_seconds'] = driver_laps_filter['LapTime'].dt.total_seconds()
    driver_laps_filter['MovingAverage'] = driver_laps_filter['LapTime_seconds'].rolling(window=3, min_periods=1).mean()
    driver_laps_filter = driver_laps_filter.reset_index()

    fig, ax = plt.subplots(figsize=(8, 8))

    plotting.COMPOUND_COLORS['TEST_UNKNOWN'] = 'grey'

    sns.scatterplot(data=driver_laps_filter,
                    x="LapNumber",
                    y="LapTime_seconds",
                    ax=ax,
                    hue="Compound",
                    palette=plotting.COMPOUND_COLORS,
                    s=125,
                    linewidth=0,
                    legend='auto')

    ax.set_xlabel("Lap Number", font='Fira Sans', fontsize=16)
    ax.set_ylabel("Lap Time", font='Fira Sans', fontsize=16)
    plt.grid(color='w', which='major', axis='both', linestyle='--')

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
    bbox_props = dict(boxstyle="square,pad=0.4", fc="white", ec="none", lw=0)

    for lap, time in zip(driver_laps_filter['LapNumber'], driver_laps_filter['LapTime_seconds']):
        if not np.isnan(time):
            ax.annotate(f"{format_func(time, None)}",
                        (lap, time),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        font='Fira Sans',
                        fontsize=11,
                        color='red',
                        bbox=bbox_props)  # Adding the white box here
    # Create a font properties object with the desired font family and size
    font_properties = FontProperties(family='Fira Sans', size='x-large')
    plt.title(f'{driver} SPRINT PACE', font='Fira Sans', fontsize=26)
    # Set the font properties for the legend
    ax.legend(prop=font_properties)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(f"../PNGs/{driver} LAPS {race.event.OfficialEventName}.png", dpi=150)
    plt.show()


def driver_race_times_per_tyre(race, driver):

    """
         Plots all the time laps, with the compound, for a driver

         Parameters:
         race (Session): Session to analyze
         driver (str): Driver

    """

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

    ax.set_xlabel("Lap Number", font='Fira Sans', fontsize=16)
    ax.set_ylabel("Lap Time", font='Fira Sans', fontsize=16)

    # The y-axis increases from bottom to top by default
    # Since we are plotting time, it makes sense to invert the axis
    ax.invert_yaxis()
    plt.suptitle(f"{driver} Laptimes in {race.event.OfficialEventName}", font='Fira Sans', fontsize=16)

    # Turn on major grid lines
    plt.grid(color='w', which='major', axis='both')
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(f"../PNGs/RACE LAPS {driver} {race.event.OfficialEventName}.png", dpi=400)
    plt.show()


def tyre_strategies(session):

    """
         Plots the tyre strategy in a race

         Parameters:
         session (Session): Session to analyze

    """

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
        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            if row['FreshTyre']:
                alpha = 1
                color = plotting.COMPOUND_COLORS[row["Compound"]]
            else:
                alpha = 0.65
                rgb_color = mcolors.to_rgb(plotting.COMPOUND_COLORS[row["Compound"]])
                color = tuple([x * 0.95 for x in rgb_color])

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

    fig.suptitle(session.event.OfficialEventName, font='Fira Sans', fontsize=14)
    plt.xlabel("Lap Number")
    plt.grid(False)
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"../PNGs/TYRE STRATEGY {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def race_pace_top_10(race):
    """
    Plots the race pace of the top 10 drivers in a race

    Parameters:
    race (session): Race to be plotted

    """
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

    point_finishers = race.drivers[:10]
    driver_laps = race.laps.pick_drivers(point_finishers).pick_quicklaps()
    driver_laps = driver_laps.reset_index()
    finishing_order = [race.get_driver(i)["Abbreviation"] for i in point_finishers]

    dict = fastf1.plotting.DRIVER_COLORS
    driver_colors = {abv: fastf1.plotting.DRIVER_COLORS[driver] for
                     abv, driver in fastf1.plotting.DRIVER_TRANSLATE.items()}

    fig, ax = plt.subplots(figsize=(8, 6))
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
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlabel("Driver", font='Fira Sans', fontsize=17)
    ax.set_ylabel("Lap Time (s)", font='Fira Sans', fontsize=17)
    plt.suptitle(f"{race.event['EventDate'].year} {race.event['EventName']} Lap Time Distributions",
                 font='Fira Sans', fontsize=20)
    sns.despine(left=False, bottom=False)
    plt.legend(title='Tyre Compound', loc='lower right', fontsize='medium')

    plt.savefig(f"../PNGs/{race.event['EventDate'].year} {race.event['EventName']} Lap Time Distributions.png", dpi=400)
    plt.tight_layout()
    plt.show()


def race_diff(team_1, team_2, year):

    """
         Plots the race time diff between 2 different teams

         Parameters:
         team_1 (str): Team 1
         team_2 (str): Team 2
         year (int): Year to analyze

    """


    session_names = []
    delta_laps = []
    colors = []
    n_races = Ergast().get_race_results(season=year, limit=1000)
    for i in range(len(n_races.content)):
        race = fastf1.get_session(year, i + 1, 'R')
        race.load(telemetry=True)
        session_names.append(race.event['Location'].split('-')[0])
        try:
            from src.utils.utils import call_function_from_module
            call_function_from_module(race_diff_team_exceptions,f"{team_1.replace(' ', '_')}_{2023}", i + 1)
            call_function_from_module(race_diff_team_exceptions,f"{team_2.replace(' ', '_')}_{2023}", i + 1)

            drivers_team_1 = list(race.laps.pick_team(team_1)['Driver'].unique())
            drivers_team_2 = list(race.laps.pick_team(team_2)['Driver'].unique())

            if team_1 == 'Ferrari' and i == 16:
                drivers_team_1.append('SAI')

            if team_1 == 'Aston Martin' and i == 14:
                drivers_team_1.append('STR')

            pace_t1_1, pace_t1_2 = get_driver_race_pace(race, drivers_team_1[0], drivers_team_1[1], team_1, i + 1,
                                                        True, True, True, 'max')
            pace_t2_1, pace_t2_2 = get_driver_race_pace(race, drivers_team_2[0], drivers_team_2[1], team_2, i + 1,
                                                        True, True, True, 'max')


            def get_driver_pace(d1, d2, pace_d1, pace_d2):
                if pace_d1 > pace_d2:
                    return d2
                else:
                    return d1
            d_t1 = get_driver_pace(*drivers_team_1, pace_t1_1, pace_t1_2)
            d_t2 = get_driver_pace(*drivers_team_2, pace_t2_1, pace_t2_2)

            pace_t1, pace_t2 = get_driver_race_pace(race, d_t1, d_t2, '', i + 1, exceptions=False)

            if pace_t1 > pace_t2:
                colors.append('#0000FF')
            else:
                colors.append('#FFA500')

            delta_diff = ((pace_t2 - pace_t1) / pace_t1) * 100
            if round(delta_diff, 2) == 0:
                if pace_t1 > pace_t2:
                    delta_laps.append(-0.01)
                else:
                    delta_laps.append(0.01)
            else:
                delta_laps.append(round(delta_diff, 2))
        except AttributeError:
            print('Cant compare')
            delta_laps.append(0)
            colors.append('#0000FF')

    fig, ax1 = plt.subplots(figsize=(10, 8))
    plt.rcParams["font.family"] = "Fira Sans"
    delta_laps = [x if not math.isnan(x) else 0 for x in delta_laps]

    print(f'MEAN: {statistics.mean(delta_laps)}')
    print(f'MEDIAN: {statistics.median(delta_laps)}')

    bars = plt.bar(session_names, delta_laps, color=colors)
    round_bars(bars, ax1, colors, color_1=plotting.team_color(team_1), color_2=plotting.team_color(team_2))
    annotate_bars(bars, ax1, 0.01, 14, text_annotate='{height}%', ceil_values=False)

    delta_laps = pd.Series(delta_laps)

    if min(delta_laps) < 0:
        plt.axhline(0, color='white', linewidth=2)

    legend_lines = [Line2D([0], [0], color=plotting.team_color(team_1), lw=4),
                    Line2D([0], [0], color=plotting.team_color(team_2), lw=4)]

    plt.legend(legend_lines, [f'{team_1} faster', f'{team_2} faster', 'Moving Average (4 last races)'],
               loc='lower left', fontsize='large')
    # Set the labels and title
    plt.ylabel(f'Percentage time difference', font='Fira Sans', fontsize=16)
    plt.xlabel('Circuit', font='Fira Sans', fontsize=16)
    ax1.yaxis.grid(True, linestyle='--')
    plt.xticks(rotation=90, fontsize=12, fontname='Fira Sans')
    plt.yticks(fontsize=12, fontname='Fira Sans')
    plt.tight_layout()
    plt.savefig(f"../PNGs/{team_2} VS {team_1} {year} race time difference.png", dpi=400)
    plt.show()


def race_distance(session, driver_1, driver_2):
    """
         Plots the race time diff between 2 drivers, each lap

         Parameters:
         session (Session): Session to analyze
         driver_1 (str): Driver 1
         driver_2 (str): Driver 2
    """

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
        laps.append(i + 1)

    laps_diff = [0 if math.isnan(x) else x for x in laps_diff]
    progressive_sum = laps_diff
    colors = ['red']
    for i in range(len(progressive_sum) - 1):
        if progressive_sum[i] < progressive_sum[i + 1]:
            colors.append('green')
        else:
            colors.append('red')
    fig, ax = plt.subplots(figsize=(37, 8))
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
            fontsize=8  # font size
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
            text_y = -16
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
    plt.title(
        f'Progressive Time Difference between {driver_1} and {driver_2} in {session.event["EventName"] + " " + str(session.event["EventDate"].year)}',
        fontsize=20)
    plt.grid(True, axis='y')
    ax.set_yticks([20, 15, 10, 5, 0, -5, -10, -15, -20])

    # Display the plot
    plt.tight_layout()  # Adjusts plot parameters for a better layout
    plt.savefig(
        f'../PNGs/Progressive Time Difference between {driver_1} and {driver_2} in {session.event["EventName"] + " " + str(session.event["EventDate"].year)}',
        dpi=400)
    plt.show()
