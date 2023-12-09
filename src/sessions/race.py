import statistics

from tabulate import tabulate

from src.exceptions import race_same_team_exceptions
from src.exceptions.custom_exceptions import RaceException
from src.plots.plots import get_font_properties
import fastf1
from fastf1.ergast import Ergast
from fastf1 import plotting

from matplotlib import pyplot as plt, ticker, patheffects
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from src.utils.utils import append_duplicate_number
from src.variables.driver_colors import driver_colors_2023
from src.variables.team_colors import team_colors_2023


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


def driver_race_times_per_tyre(session, driver):
    """
         Plots all the time laps, with the compound, for a driver

         Parameters:
         race (Session): Session to analyze
         driver (str): Driver

    """

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    driver_laps = session.laps.pick_driver(driver).pick_quicklaps().reset_index()
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

    plt.suptitle(f"{driver} Laptimes in {session.event.OfficialEventName}", font='Fira Sans', fontsize=14)

    # Turn on major grid lines
    plt.grid(color='w', which='major', axis='both')
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(f"../PNGs/RACE LAPS {driver} {session.event.OfficialEventName}.png", dpi=500)
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
        if past_stint != current_stint and i != 0:
            tyre_laps_prev = stints.loc[i - 1, 'TyreLife']
            tyre_laps = stints.loc[i, 'TyreLife']
            compound_prev = stints.loc[i - 1, 'Compound']
            compound = stints.loc[i, 'Compound']

            if tyre_laps_prev + 1 == tyre_laps and (compound_prev == compound):
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

    fig, ax = plt.subplots(figsize=(8, 8))

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
            stint_text = str(row["StintLength"])
            text_x_position = previous_stint_end + row["StintLength"] / 2  # center the text in the stint bar
            text_y_position = driver  # align vertically with the driver's bar
            text = ax.text(
                text_x_position, text_y_position, stint_text,
                ha='center', va='center', font='Fira Sans', fontsize=12, color='black', alpha=0.75
            )

            text.set_path_effects([
                patheffects.withStroke(linewidth=0.8, foreground="white")
            ])

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
    plt.legend(handles=patches.values(), bbox_to_anchor=(0.5, -0.1),
               prop=get_font_properties('Fira Sans', 'large'), loc='upper center', ncol=2)

    plt.title(session.event.OfficialEventName, font='Fira Sans', fontsize=18)
    plt.xlabel("Lap Number", font='Fira Sans')
    plt.grid(False)
    ax.invert_yaxis()

    plt.xticks(font='Fira Sans', fontsize=14)
    plt.yticks(font='Fira Sans', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"../PNGs/TYRE STRATEGY {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def race_pace_top_10(session, threshold=1.07):
    """
    Plots the race pace of the top 10 drivers in a race

    Parameters:
    race (session): Race to be plotted

    """
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

    point_finishers = session.drivers[:10]
    point_finishers = ['VER', 'RUS', 'LEC', 'NOR', 'ALO', 'TSU', 'GAS', 'ALB', 'HUL', 'ZHO']
    driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps(threshold).pick_wo_box()
    driver_laps = driver_laps.reset_index()
    finishing_order = [session.get_driver(i)["Abbreviation"] for i in point_finishers]
    driver_colors = {abv: fastf1.plotting.DRIVER_COLORS[driver] for
                     abv, driver in fastf1.plotting.DRIVER_TRANSLATE.items()}

    fig, ax = plt.subplots(figsize=(8, 6))
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()
    mean_lap_times = driver_laps.groupby("Driver")["LapTime(s)"].mean()
    median_lap_times = driver_laps.groupby("Driver")["LapTime(s)"].median()

    # driver_colors = ['#0600ef', '#dc0000', '#ff8700', '#00d2be', '#FF69B4', '#006f62',
    #                  '#005aff', '#cacaca', '#2b4562', '#900000']

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
        # ax.text(i, min(driver_laps['LapTime(s)']) - 1.5,
        #         f"{int(mean_minutes):02d}:{int(mean_seconds):02d}.{int(mean_milliseconds * 1000):03d}",
        #         font='Fira Sans', fontsize=9, ha='center')
        ax.text(i, min(driver_laps['LapTime(s)']) - 1.5,
                f"{int(median_minutes):02d}:{int(median_seconds):02d}.{int(median_milliseconds * 1000):03d}",
                font='Fira Sans', fontsize=9, ha='center')

    plt.ylim(min(driver_laps['LapTime']).total_seconds() - 2,
             max(driver_laps['LapTime']).total_seconds() + 2)  # change these numbers as per your needs
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlabel("Driver", font='Fira Sans', fontsize=17)
    ax.set_ylabel("Lap Time (s)", font='Fira Sans', fontsize=17)
    plt.suptitle(f"{session.event['EventDate'].year} {session.event['EventName']} Lap Time Distributions",
                 font='Fira Sans', fontsize=20)
    sns.despine(left=False, bottom=False)
    plt.legend(title='Tyre Compound', loc='upper left', fontsize='medium')

    plt.tight_layout()
    plt.savefig(f"../PNGs/{session.event['EventDate'].year} {session.event['EventName']} Lap Time Distributions.png",
                dpi=450)
    plt.show()


def race_diff(year):
    """
         Plots the race time diff between 2 different teams

         Parameters:
         year (int): Year to analyze

    """

    session_names = []
    n_races = Ergast().get_race_results(season=year, limit=1000)
    mean_delta_by_team = {}
    for i in range(len(n_races.content)):
        if not (year == 2021 and (i + 1) == 12):
            race = fastf1.get_session(year, i + 1, 'R')
            race.load(telemetry=True)
            total_laps = race.total_laps
            session_names.append(race.event['Location'].split('-')[0])
            teams = race.results['TeamName'].unique()
            teams = teams[teams != 'nan']
            pace_with_laps = {}
            for t in teams:
                drivers = race.laps[race.laps['Team'] == t]['Driver'].unique()
                n_laps = [len(race.laps[race.laps['Driver'] == i]) for i in drivers]
                if race.event.Location == 'Monaco' and year == 2022 and t == 'McLaren':
                    n_laps = [51]
                if race.event.Location == 'Imola' and year == 2022 and t == 'Ferrari':
                    n_laps = [49]
                paces = {}
                for d in drivers:
                    for n in set(n_laps):
                        if n >= total_laps / 3:
                            if len(race.laps.pick_driver(d)[0:n]) >= n:
                                mean = race.laps.pick_driver(d)[0:n].pick_quicklaps().pick_wo_box()['LapTime'].mean()
                                if d not in paces:
                                    paces[d] = {'MEAN': [mean],
                                                'LAPS': [n]}
                                else:
                                    paces[d]['MEAN'].append(mean)
                                    paces[d]['LAPS'].append(n)
                if len(paces) == 0:
                    if t not in mean_delta_by_team:
                        mean_delta_by_team[t] = [np.NaN]
                    else:
                        mean_delta_by_team[t].append(np.NaN)
                else:
                    if race.event.Location == 'Spielberg' and year == 2022 and t == 'Red Bull Racing':
                        paces.pop('PER')
                    if len(paces) > 1:
                        paces = dict(sorted(paces.items(),
                                            key=lambda item: (min(item[1]["LAPS"]),
                                                              item[1]["MEAN"][
                                                                  item[1]["LAPS"].index(min(item[1]["LAPS"]))])))
                    else:
                        key = next(iter(paces))  # Get the single key
                        laps, means = paces[key]["LAPS"], paces[key]["MEAN"]
                        sorted_pairs = sorted(zip(laps, means), key=lambda pair: pair[1])  # Sort by "MEAN"
                        sorted_laps, sorted_means = zip(*sorted_pairs)  # Unzip the sorted pairs
                        paces = {key: {"LAPS": sorted_laps, "MEAN": sorted_means}}
                    pace_with_laps[t] = {"MEAN": paces[list(paces.keys())[0]]['MEAN'][0],
                                         "LAPS": paces[list(paces.keys())[0]]['LAPS'][0]}
            pace_with_laps = dict(sorted(pace_with_laps.items(), key=lambda item: (item[1]["MEAN"])))
            current_fastest_team = list(pace_with_laps.keys())[0]
            current_fastest_mean = pace_with_laps[list(pace_with_laps.keys())[0]]['MEAN']
            pace_with_laps = dict(sorted(pace_with_laps.items(), key=lambda item: (-item[1]["LAPS"], item[1]["MEAN"])))
            for t, m in pace_with_laps.items():
                if m['LAPS'] < total_laps - 5:
                    laps = m['LAPS']
                    pace = m['MEAN']
                    drivers_fastest_team = race.laps[race.laps['Team'] == current_fastest_team]['Driver'].unique()
                    mean_fixed_laps = []
                    for d in drivers_fastest_team:
                        mean_fixed_laps.append(
                            race.laps.pick_driver(d)[0:laps].pick_quicklaps().pick_wo_box()['LapTime'].mean())
                    fastest_fixed_mean = min(mean_fixed_laps)
                    delta_diff = ((pace - fastest_fixed_mean) / fastest_fixed_mean) * 100
                    if t not in mean_delta_by_team:
                        mean_delta_by_team[t] = [delta_diff]
                    else:
                        mean_delta_by_team[t].append(delta_diff)
                elif t != current_fastest_team:
                    team_mean = m['MEAN']
                    delta_diff = ((team_mean - current_fastest_mean) / current_fastest_mean) * 100
                    if t not in mean_delta_by_team:
                        mean_delta_by_team[t] = [delta_diff]
                    else:
                        mean_delta_by_team[t].append(delta_diff)
                elif t == current_fastest_team:
                    if t not in mean_delta_by_team:
                        mean_delta_by_team[t] = [0]
                    else:
                        mean_delta_by_team[t].append(0)
            min_latest_value = min(value[-1] if isinstance(value, list) and len(value) > 0 else 0
                                   for value in mean_delta_by_team.values())
            if min_latest_value != 0:
                for team, value in mean_delta_by_team.items():
                    if len(value) > 0:
                        mean_delta_by_team[team][-1] = value[-1] - min_latest_value
                print(f'Recompute {race.event.Location}')
        else:
            print('THE INFAMOUS')

    for t, d in mean_delta_by_team.items():
        print(f'{t} - {d}')

    session_names = append_duplicate_number(session_names)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.rcParams["font.family"] = "Fira Sans"
    for team, deltas in mean_delta_by_team.items():
        deltas_array = np.array(deltas)
        not_nan_indices = ~np.isnan(deltas_array)
        plt.plot(np.array(session_names)[not_nan_indices], deltas_array[not_nan_indices],
                 label=team, marker='o', color=team_colors_2023.get(team), markersize=7, linewidth=3)
    plt.gca().invert_yaxis()
    plt.legend(loc='lower right', fontsize='medium')
    plt.title(f'{year} AVERAGE RACE PACE PER CIRCUIT', font='Fira Sans', fontsize=20)
    plt.ylabel('Percentage time difference (%)', font='Fira Sans', fontsize=16)
    plt.xlabel('Circuit', font='Fira Sans', fontsize=16)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(True, linestyle='--', alpha=0.2)
    plt.xticks(rotation=90, fontsize=12, fontname='Fira Sans')
    plt.yticks(fontsize=12, fontname='Fira Sans')
    plt.tight_layout()
    plt.savefig(f"../PNGs/{year} race time difference.png", dpi=400)
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
    bar_colors = [driver_colors_2023[d] for d in drivers]
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
    for c in ['MEDIAN', 'MEAN']:
        print(f'{c} DIFFERENCE')
        for t in teams:
            drivers = session.laps.pick_team(t)['Driver'].unique()
            d1_laps = session.laps.pick_driver(drivers[0])
            d2_laps = session.laps.pick_driver(drivers[1])

            if len(d1_laps) > 10 and len(d2_laps) > 10:

                comp_laps = min(len(d1_laps), len(d2_laps))
                d1_laps = session.laps.pick_driver(drivers[0])[0:comp_laps].pick_quicklaps().pick_wo_box()
                d2_laps = session.laps.pick_driver(drivers[1])[0:comp_laps].pick_quicklaps().pick_wo_box()
                if c == 'MEDIAN':
                    d1_time = d1_laps['LapTime'].median()
                    d2_time = d2_laps['LapTime'].median()
                else:
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


def race_pace_between_drivers(year, d1, d2):
    schedule = fastf1.get_event_schedule(2023, include_testing=False)
    total_laps_d1 = []
    total_laps_d2 = []
    for i in range(len(schedule)):
        session = fastf1.get_session(year, i + 1, 'R')
        session.load()
        try:
            team_d1 = session.laps.pick_driver(d1)['Team'].unique()[0]
            team_d2 = session.laps.pick_driver(d2)['Team'].unique()[0]
            from src.utils.utils import call_function_from_module
            min_laps_d1, max_laps_d1 = call_function_from_module(race_same_team_exceptions,
                                                           f"{team_d1.replace(' ', '_')}_{year}",
                                                           i + 1, session.total_laps)
            min_laps_d2, max_laps_d2 = call_function_from_module(race_same_team_exceptions,
                                                           f"{team_d2.replace(' ', '_')}_{year}",
                                                           i + 1, session.total_laps)
            min_laps = max(min_laps_d1, min_laps_d2)
            max_laps = min (max_laps_d1, max_laps_d2)
            d1_laps = session.laps.pick_driver(d1)[min_laps:max_laps].pick_quicklaps().pick_wo_box()
            d2_laps = session.laps.pick_driver(d2)[min_laps:max_laps].pick_quicklaps().pick_wo_box()
            d1_compare = pd.DataFrame(d1_laps[['LapNumber', 'LapTime', 'Compound', 'TyreLife']])
            d2_compare = pd.DataFrame(d2_laps[['LapNumber', 'LapTime', 'Compound', 'TyreLife']])
            d1_compare.columns = ['LapNumber', 'LapTime_d1', 'Compound_d1', 'TyreLife_d1']
            d2_compare.columns = ['LapNumber', 'LapTime_d2', 'Compound_d2', 'TyreLife_d2']
            comparable_laps = pd.merge(d1_compare, d2_compare, how='inner', on='LapNumber')
            comparable_laps = comparable_laps[comparable_laps['Compound_d1'] == comparable_laps['Compound_d2']]
            comparable_laps = comparable_laps[abs(comparable_laps['TyreLife_d1'] - comparable_laps['TyreLife_d2']) <= 3]
            if len(comparable_laps) > 5:
                print(tabulate(comparable_laps, headers='keys', tablefmt='fancy_grid'))
                total_laps_d1.extend(comparable_laps['LapTime_d1'].values)
                total_laps_d2.extend(comparable_laps['LapTime_d2'].values)
        except (RaceException, IndexError):
            print(f'{session}')
    race_differences = []
    for l1, l2 in zip(total_laps_d1, total_laps_d2):
        delta_diff = (l1 - l2) / ((l1 + l2) / 2) * 100
        race_differences.append(delta_diff)

    median = statistics.median(race_differences)
    mean = statistics.mean(race_differences)
    print(race_differences)
    print(f'MEDIAN DIFF {d1 if median < 0 else d2} FASTER:{median}')
    print(f'MEAN DIFF {d1 if mean < 0 else d2} FASTER:{mean}')
