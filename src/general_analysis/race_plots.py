import re
import statistics

from adjustText import adjust_text
from matplotlib.font_manager import FontProperties

from src.plots.plots import round_bars, annotate_bars, get_font_properties, lighten_color
from src.exceptions import race_same_team_exceptions, qualy_exceptions, race_diff_team_exceptions
import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from fastf1 import plotting

from matplotlib import pyplot as plt, patches, image as mpimg, ticker
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


def long_runs_FP2(race, driver, threshold=1.07):
    """
         Plots the long runs in FP2 for a given driver

         Parameters:
         race (Session): Session to analyze
         driver (str): Driver

    """

    plotting.setup_mpl(misc_mpl_mods=False)

    driver_laps = race.laps.pick_driver(driver)
    max_stint = driver_laps['Stint'].value_counts().index[0]
    driver_laps_filter = driver_laps[driver_laps['Stint'] == max_stint].pick_quicklaps(threshold)

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


    bbox_props = dict(boxstyle="square,pad=0.4", fc="white", ec="none", lw=0)
    annotations = []
    for lap, time in zip(driver_laps_filter['LapNumber'], driver_laps_filter['LapTime_seconds']):
        if not np.isnan(time):
            text = ax.annotate(f"{format_func(time, None)}",
                               (lap, time),
                               textcoords="offset points",
                               xytext=(0, 10),
                               ha='center',
                               fontsize=11,
                               color='red',
                               bbox=bbox_props)
            annotations.append(text)

    adjust_text(annotations, ax=ax, autoalign='xy')

    font_properties = FontProperties(family='Fira Sans', size='x-large')
    plt.title(f'{driver} LONG RUNS IN {str(race.event.year) + " " + race.event.Country + " " + race.name}',
              font='Fira Sans', fontsize=20)
    # Set the font properties for the legend
    ax.legend(prop=font_properties, loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    sns.despine(left=True, bottom=True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.figtext(0.01, 0.02, '@Big_Data_Master', font='Fira Sans', fontsize=17, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../PNGs/{driver} LAPS {race.event.OfficialEventName}.png", dpi=450)
    plt.show()



def long_runs_FP2_v2(race, threshold=1.07):
    """
         Plots the long runs in FP2 for a given driver

         Parameters:
         race (Session): Session to analyze
         driver (str): Driver

    """

    plotting.setup_mpl(misc_mpl_mods=False)
    drivers = list(race.laps['Driver'].unique())
    driver_laps = pd.DataFrame(columns=['Driver', 'Laps', 'Compound', 'Average'])
    for d in drivers:
        d_laps = race.laps.pick_driver(d)
        max_stint = d_laps['Stint'].value_counts().index[0]
        driver_laps_filter = d_laps[d_laps['Stint'] == max_stint].pick_quicklaps(threshold).pick_wo_box()
        stint_index = 1
        try:
            while len(driver_laps_filter) < 5:
                max_stint = d_laps['Stint'].value_counts().index[stint_index]
                driver_laps_filter = d_laps[d_laps['Stint'] == max_stint].pick_quicklaps(threshold).pick_wo_box()
                stint_index += 1
            driver_laps_filter = driver_laps_filter[driver_laps_filter['LapTime'].notna()]
            driver_laps_filter['LapNumber'] = driver_laps_filter['LapNumber'] - driver_laps_filter['LapNumber'].min() + 1
            driver_laps_filter = driver_laps_filter.reset_index()
            df_append = pd.DataFrame({
                'Driver': d,
                'Laps': [driver_laps_filter['LapTime'].to_list()],
                'Compound': [driver_laps_filter['Compound'].iloc[0]],
                'Average': driver_laps_filter['LapTime'].mean()
            })
            driver_laps = pd.concat([driver_laps, df_append], ignore_index=True)
        except:
            print(f'NO DATA FOR {d}')

    driver_laps = driver_laps.sort_values(by=['Compound', 'Average'], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(8, 8))

    def darken_color_custom(color, amount=0.5):
        """
        Darkens the given color by reducing each RGB component by a given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> darken_color('g', 0.3)
        >> darken_color('#F034A3', 0.6)
        >> darken_color((.3,.55,.1), 0.5)
        """
        try:
            c = mcolors.cnames[color]
        except:
            c = color
        c = list(mcolors.to_rgb(c))
        c = [max(0, i - i * amount) for i in c]
        return c

    for idx, row in driver_laps.iterrows():
        driver = row['Driver']
        laps = row['Laps']
        tyre = row['Compound']
        hex_color = plotting.COMPOUND_COLORS[tyre]
        color_factor = np.linspace(0, 0.6, len(laps))
        color_index = 0
        for lap in laps:
            color = darken_color_custom(hex_color, amount=round(color_factor[color_index], 1))
            plt.scatter(lap, driver, color=color, s=80)
            color_index += 1

    ax.set_xlabel("Lap Time", font='Fira Sans', fontsize=16)
    ax.set_ylabel("Driver", font='Fira Sans', fontsize=16)
    plt.grid(color='w', which='major', axis='x', linestyle='--')
    plt.xticks(font='Fira Sans', fontsize=14)
    plt.yticks(font='Fira Sans', fontsize=14)
    font_properties = FontProperties(family='Fira Sans', size='large')
    plt.title(f'STINTS IN {str(race.event.year) + " " + race.event.Country + " " + race.name}',
              font='Fira Sans', fontsize=20)
    soft_patch = mpatches.Patch(color='red', label='Soft')
    medium_patch = mpatches.Patch(color='yellow', label='Medium')
    hard_patch = mpatches.Patch(color='lightgray', label='Hard')
    ax.legend(handles=[soft_patch, medium_patch, hard_patch], prop=font_properties, loc='upper right')
    sns.despine(left=True, bottom=True)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', font='Fira Sans', fontsize=17, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../PNGs/ LAPS {race.event.OfficialEventName}.png", dpi=450)
    plt.show()




def driver_race_times_per_tyre(race, driver):

    """
         Plots all the time laps, with the compound, for a driver

         Parameters:
         race (Session): Session to analyze
         driver (str): Driver

    """

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    driver_laps = race.laps.pick_driver(driver).pick_quicklaps().reset_index()
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(data=driver_laps,
                    x="LapNumber",
                    y="LapTime(s)",
                    ax=ax,
                    hue="Compound",
                    palette=fastf1.plotting.COMPOUND_COLORS,
                    s=80,
                    linewidth=0,
                    legend='auto')

    ax.set_xlabel("Lap Number", font='Fira Sans', fontsize=16)
    ax.set_ylabel("Lap Time", font='Fira Sans', fontsize=16)
    def lap_time_formatter(x, pos):
        minutes, seconds = divmod(x, 60)
        return f"{int(minutes):02}:{seconds:06.3f}"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lap_time_formatter))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(f"{driver} Laptimes in {race.event.OfficialEventName}", font='Fira Sans', fontsize=14)

    # Turn on major grid lines
    plt.grid(color='w', which='major', axis='both')
    sns.despine(left=True, bottom=True)


    plt.tight_layout()
    plt.savefig(f"../PNGs/RACE LAPS {driver} {race.event.OfficialEventName}.png", dpi=500)
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

    fig.suptitle(session.event.OfficialEventName, font='Fira Sans', fontsize=12)
    plt.xlabel("Lap Number")
    plt.grid(False)
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"../PNGs/TYRE STRATEGY {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def race_pace_top_10(race, threshold=1.07):
    """
    Plots the race pace of the top 10 drivers in a race

    Parameters:
    race (session): Race to be plotted

    """
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

    point_finishers = race.drivers[:10]
    point_finishers = ['VER', 'NOR', 'ALO', 'SAI', 'GAS', 'HAM', 'TSU', 'SAR', 'HUL']
    driver_laps = race.laps.pick_drivers(point_finishers).pick_quicklaps(threshold).pick_wo_box()
    driver_laps = driver_laps.reset_index()
    finishing_order = [race.get_driver(i)["Abbreviation"] for i in point_finishers]
    driver_colors = {abv: fastf1.plotting.DRIVER_COLORS[driver] for
                     abv, driver in fastf1.plotting.DRIVER_TRANSLATE.items()}

    fig, ax = plt.subplots(figsize=(8, 6))
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()
    mean_lap_times = driver_laps.groupby("Driver")["LapTime(s)"].mean()
    median_lap_times = driver_laps.groupby("Driver")["LapTime(s)"].median()

    # driver_colors = ['#00d2be', '#dc0000', '#FF69B4', '#0600ef', '#900000', '#006f62',
    #                  '#ffffff', '#2b4562', '#005aff']

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

    for i, driver in enumerate(finishing_order):
        mean_time = mean_lap_times[driver]
        median_time = median_lap_times[driver]

        # Convert mean and median times to minutes, seconds, and milliseconds
        mean_minutes, mean_seconds = divmod(mean_time, 60)
        mean_seconds, mean_milliseconds = divmod(mean_seconds, 1)
        median_minutes, median_seconds = divmod(median_time, 60)
        median_seconds, median_milliseconds = divmod(median_seconds, 1)

        # Place the mean and median labels below each violinplot
        # ax.text(i, min(driver_laps['LapTime(s)']) - 1.3,
        #         f"Average",
        #         font='Fira Sans', fontsize=9, ha='center')
        ax.text(i, min(driver_laps['LapTime(s)']) - 1.5,
                f"{int(mean_minutes):02d}:{int(mean_seconds):02d}.{int(mean_milliseconds * 1000):03d}",
                font='Fira Sans', fontsize=9, ha='center')
        # ax.text(i, min(driver_laps['LapTime(s)']) - 1.5,
        #         f"{int(median_minutes):02d}:{int(median_seconds):02d}.{int(median_milliseconds * 1000):03d}",
        #         font='Fira Sans', fontsize=9, ha='center')

    plt.ylim(min(driver_laps['LapTime']).total_seconds() - 2,
             max(driver_laps['LapTime']).total_seconds() + 2)  # change these numbers as per your needs
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlabel("Driver", font='Fira Sans', fontsize=17)
    ax.set_ylabel("Lap Time (s)", font='Fira Sans', fontsize=17)
    plt.suptitle(f"{race.event['EventDate'].year} {race.event['EventName']} Lap Time Distributions",
                 font='Fira Sans', fontsize=20)
    sns.despine(left=False, bottom=False)
    plt.legend(title='Tyre Compound', loc='upper left', fontsize='medium')

    plt.savefig(f"../PNGs/{race.event['EventDate'].year} {race.event['EventName']} Lap Time Distributions.png", dpi=450)
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


def race_distance(session, top=10):
    """
         Plots the race time diff between 2 drivers, each lap

         Parameters:
         session (Session): Session to analyze
         driver_1 (str): Driver 1
         driver_2 (str): Driver 2
    """

    drivers = session.results['Abbreviation'].values[:top]
    laps = session.total_laps
    diff = [[] for _ in range(len(drivers))]
    leader_laps = session.laps.pick_driver(drivers[0])
    leader_laps = leader_laps['Time'].dt.total_seconds().reset_index(drop=True)
    driver_colors = {abv: fastf1.plotting.DRIVER_COLORS[driver] for
                     abv, driver in fastf1.plotting.DRIVER_TRANSLATE.items()}
    bar_colors = [driver_colors[d] for d in drivers]
    laps_array = [i for i in range(1, laps + 1)]
    sc_laps = session.laps.pick_driver(drivers[0])['TrackStatus']
    sc_laps = sc_laps.astype(str).str.contains('4|5|6|7')
    s_int = sc_laps.astype(int)
    cumsum = s_int.groupby((s_int != s_int.shift()).cumsum()).cumsum()
    sc_laps = cumsum[cumsum > 0].index.values + 1
    for i in range(len(drivers)):
        diff[i] = []
        d_laps = session.laps.pick_driver(drivers[i])
        d_laps = d_laps['Time'].dt.total_seconds().reset_index(drop=True)
        if i == 0:
            diff[i].extend([0] * laps)
        else:
            delta_diff = d_laps - leader_laps
            diff[i].extend(delta_diff.values)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(len(diff)):
        plt.plot(laps_array, diff[i], color=bar_colors[i], label=drivers[i])
    sc_intervals = zip(sc_laps, sc_laps[1:] if len(sc_laps) > 1 else sc_laps)
    for start, end in sc_intervals:
        ax.axvspan(start, end, color='yellow', alpha=0.3)

    plt.gca().invert_yaxis()
    plt.xlabel('Laps', font='Fira Sans', fontsize=16)
    plt.ylabel('Progressive Time Difference (seconds)', font='Fira Sans', fontsize=16)
    plt.xticks(font='Fira Sans', fontsize=14)
    plt.yticks(font='Fira Sans', fontsize=14)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', font='Fira Sans', fontsize=15, color='gray', alpha=0.5)
    plt.title(
        f'Progressive Time Difference between in {session.event["EventName"] + " " + str(session.event["EventDate"].year)}',
        font='Fira Sans', fontsize=17)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(prop=get_font_properties('Fira Sans', 12), loc='lower left', fontsize='large')
    plt.tight_layout()  # Adjusts plot parameters for a better layout
    plt.savefig(
        f'../PNGs/Progressive Time Difference between in {session.event["EventName"] + " " + str(session.event["EventDate"].year)}',
        dpi=400)
    plt.show()


def teams_diff_session(session):

    teams = session.laps['Team'].unique()
    for t in teams:
        drivers = session.laps.pick_team(t)['Driver'].unique()
        d1_laps = session.laps.pick_driver(drivers[0])
        d2_laps = session.laps.pick_driver(drivers[1])

        if len(d1_laps) > 10 and len(d2_laps) > 10:

            comp_laps = min(len(d1_laps), len(d2_laps))
            d1_laps = session.laps.pick_driver(drivers[0])[0:comp_laps].pick_quicklaps().pick_wo_box()
            d2_laps = session.laps.pick_driver(drivers[1])[0:comp_laps].pick_quicklaps().pick_wo_box()
            d1_time = d1_laps['LapTime'].mean()
            d2_time = d2_laps['LapTime'].mean()
            diff = round((d2_time - d1_time).total_seconds(), 3)

            def format_time_delta(td):
                minutes = (td.seconds // 60) % 60
                seconds = td.seconds % 60
                milliseconds = td.microseconds // 1000
                return f'{minutes}:{seconds}.{milliseconds}'

            if diff < 0:
                print(f'{drivers[1]}: {format_time_delta(d2_time)}\n'
                      f'{drivers[0]}: {format_time_delta(d1_time)} (+{-diff})')
            else:
                print(f'{drivers[0]}: {format_time_delta(d1_time)}\n'
                      f'{drivers[1]}: {format_time_delta(d2_time)} (+{diff})')