import fastf1
import numpy as np
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt, cm
import re
from matplotlib.lines import Line2D

from src.ergast_api.my_ergast import My_Ergast
from src.plots.plots import stacked_bars, round_bars, annotate_bars
from src.variables.variables import point_systems


def win_wdc(year):
    """
         Plots who can win the WDC in a year

         Parameters:
         year (int): Year to plot
    """
    ergast = Ergast()
    standings = ergast.get_driver_standings(season=year)
    driver_standings = standings.content[0]

    POINTS_FOR_SPRINT = 8 + 25 + 1  # Winning the sprint, race and fastest lap
    POINTS_FOR_CONVENTIONAL = 25 + 1  # Winning the race and fastest lap

    events = fastf1.events.get_events_remaining(force_ergast=True)
    # Count how many sprints and conventional races are left
    sprint_events = len(events.loc[events["EventFormat"] == "sprint_shootout"])
    conventional_events = len(events.loc[events["EventFormat"] == "conventional"])

    # Calculate points for each
    sprint_points = sprint_events * POINTS_FOR_SPRINT
    conventional_points = conventional_events * POINTS_FOR_CONVENTIONAL

    points = sprint_points + conventional_points

    LEADER_POINTS = int(driver_standings.loc[0]['points'])

    for i, _ in enumerate(driver_standings.iterrows()):
        driver = driver_standings.loc[i]
        driver_max_points = int(driver["points"]) + points
        can_win = 'No' if driver_max_points < LEADER_POINTS else 'Yes'

        print(f"{driver['position']}: {driver['givenName'] + ' ' + driver['familyName']}, "
              f"Current points: {driver['points']}, "
              f"Theoretical max points: {driver_max_points}, "
              f"Can win: {can_win}")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = plt.bar(driver_standings['driverCode'], driver_standings['points'] + points, label='Maximum Points Possible')

    stacked_bars(bars, ax, "#000000")

    bars = plt.bar(driver_standings['driverCode'], driver_standings['points'], label='Current Points')

    stacked_bars(bars, ax, "#FF0000")

    plt.axhline(y=driver_standings['points'].max(), color='r', linestyle='--', label='Points cap')

    plt.xlabel('Drivers')
    plt.ylabel('Points')
    plt.title(
        f'Who can still win the WDC? - Season {standings.description.season[0]} with {len(events)} races remaining')
    legend_lines = [Line2D([0], [0], color='#008000', lw=4),
                    Line2D([0], [0], color='#FF0000', lw=4),
                    Line2D([0], [0], color='red', lw=2.5, linestyle='--')]

    plt.legend(legend_lines, ['Max possible points', 'Current points', 'Points cap'],
               loc='upper right', fontsize='large')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')

    plt.savefig(f"../PNGs/CAN WIN WDC - SEASON{standings.description.season[0]} AT {len(events)} REMAINING.png",
                dpi=400)
    # Display the plot
    plt.show()


def process_race_data(content, driver, points, team_mate_points, team_mates_names, DNFs=None):
    append = False
    for race in content:
        race_data = race[(race['familyName'] == driver.split(" ")[1]) & (race['givenName'] == driver.split(" ")[0])]
        if len(race_data) > 0:
            driver_team = race_data['constructorName'].values[0]
            if re.search(r'(Spun off|Accident|Finished|Collision|\+)', race_data['status'].max()) and DNFs is not None:
                team_mates = race[race['constructorName'] == driver_team]
                team_mates = team_mates[team_mates['familyName'] != driver.split(" ")[1]]
                if re.search(r'(Spun off|Accident|Finished|Collision|\+)', team_mates['status'].max()):
                    points += race_data['points'].values[0]
                    team_mate_points += team_mates['points'].values[0]
                    team_mates_names.append(team_mates['driverCode'].values[0])
                    append = True
            elif DNFs is None:
                team_mates = race[race['constructorName'] == driver_team]
                team_mates = team_mates[team_mates['familyName'] != driver.split(" ")[1]]
                points += race_data['points'].values[0]
                if len(team_mates) == 0:
                    team_mate_points += 0
                    team_mates_names.append("Didn't have")
                else:
                    if team_mates['familyName'].values[0] == 'Andretti':
                        a = 1
                    team_mate_points += team_mates['points'].values[0]
                    team_mates_names.append(team_mates['familyName'].values[0])
                append = True

    return points, team_mate_points, team_mates_names, append


def wdc_comparation(driver, start=None, end=None, DNFs=None):
    """
         Plots the driver given with the points comparison against all his teammates

         Parameters:
         driver (str): Driver
         start (int, optional): Year of start. Default: 1950
         end (int, optional): Year of end. Default: 2024
         DNFs (int, optional): Count DNFs. Default: None
    """

    ergast = Ergast()

    start = start if start is not None else 1950
    end = end if end is not None else 2024

    points_per_year = []
    years = []
    team_mates_points_per_year = []
    team_mates_names_per_year = []
    driver_code = []

    for i in range(start, end):

        races = ergast.get_race_results(season=i, limit=1000)
        sprints = ergast.get_sprint_results(season=i, limit=1000)
        points = 0
        team_mate_points = 0
        team_mates_names = []

        (points, team_mate_points,
         team_mates_names, append_races) = process_race_data(
            races.content, driver, points, team_mate_points, team_mates_names, DNFs)
        (points, team_mate_points,
         team_mates_names, append_sprints) = process_race_data(
            sprints.content, driver, points, team_mate_points, team_mates_names, DNFs)

        if append_races or append_sprints:
            points_per_year.append(points)
            team_mates_points_per_year.append(team_mate_points)
            team_mates_names_per_year.append(team_mates_names)
            driver_code.append(driver)
            years.append(i)
        print(i)

    fixed_names = []
    for year in team_mates_names_per_year:
        unique_names = set(year)
        formatted_elements = [f"{elem}" for elem in unique_names]
        formatted_string = ' - '.join(formatted_elements)
        fixed_names.append(formatted_string)

    points_diff = np.subtract(points_per_year, team_mates_points_per_year)

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=150)

    all_teammate_names = set()
    for names in fixed_names:
        all_teammate_names.add(names)

    color_map = cm.get_cmap('tab20', len(all_teammate_names))

    first_three_letters = [
        '/'.join(
            part[:3].upper() if '.' not in part else part.split('.')[0][:3].upper()
            for part in name.replace(' - ', '/').split('/')
        )
        for name in fixed_names
    ]
    text_to_annotate = []
    for p, d in zip(points_diff, first_three_letters):
        if p > 0:
            text = f'+{p:.0f}\nVS\n{d}'
        else:
            text = f'{p:.0f}\nVS\n{d}'
        text_to_annotate.append(text)

    bars = plt.bar(years, points_diff)
    round_bars(bars, ax1, color_map, color_1='#32CD32', color_2='#FF0000', y_offset_rounded=0, corner_radius=0.4)
    annotate_bars(bars, ax1, 2.5, 13, text_annotate=text_to_annotate, ceil_values=False,
                  round=0, y_negative_offset='height', annotate_zero=True, negative_offset=21.5)

    ax1.set_ylim(top=np.max(points_diff) + 25)
    y_min = np.min(points_diff)
    if y_min < 0:
        ax1.set_ylim(bottom=y_min * 3)
    plt.xlabel('Year', font='Fira Sans', fontsize=16)
    plt.ylabel('Points difference', font='Fira Sans', fontsize=16)

    plt.title(f'{driver} points comparison per year with his teammates {"- excluding mechanical DNFs" if DNFs else ""}',
              font='Fira Sans', fontsize=18,
              )
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.gca().xaxis.grid(False)
    ax1.set_xticks(years)
    ax1.set_xticklabels(years)
    plt.xticks(font='Fira Sans', fontsize=14, rotation=45)
    plt.yticks(font='Fira Sans', fontsize=14)

    plt.figtext(0.01, 0.02, '@Big_Data_Master', font='Fira Sans', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver} POINTS COMPARISON WDC {"DNFs" if DNFs is True else ""}.png', dpi=450)

    plt.show()


def team_wdc_history(team, color='papaya'):
    """
         Plots all the results for a team in the constructors championship

         Parameters:
         team (str): Team
         color (str, optional): Color of the line. Default: papaya
    """

    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)
    ergast = Ergast()
    years = []
    positions = []
    for i in range(1950, 2024):
        renamed_team = team
        if i in [1967] and team == 'McLaren':
            renamed_team = 'McLaren-BRM'
        elif i <= 1970 and team == 'McLaren':
            renamed_team = 'McLaren-Ford'
        if i == 1969 and team == 'McLaren':
            renamed_team = 'BRM'
        standings = ergast.get_constructor_standings(season=i, limit=1000)
        if len(standings.content) > 0:
            team_data = standings.content[0][standings.content[0]['constructorName'] == renamed_team]
            if len(team_data) > 0:
                position = team_data['position'].min()
                years.append(i)
                positions.append(position)

        print(years, positions)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(years, positions, color=color, marker='o', linewidth=3, zorder=2)

    plt.yticks(np.arange(1, max(positions) + 1, 1))
    plt.xticks(np.arange(min(years), max(years) + 1, 5), rotation=90)
    plt.gca().invert_yaxis()

    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    for year, position in zip(years, positions):
        if position == 1:
            print(f'{year}')

    plt.title(f'{team.upper()} HISTORICAL CONSTRUCTOR POSITIONS', font='Fira Sans', fontsize=18)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Constructors position', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.savefig(f'../PNGs/{team} history.png', dpi=450)
    plt.show()


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

    qualy_data = My_Ergast().get_qualy_results([year]).content
    drivers = set([code for df in qualy_data for code in df['driverCode'].values])
    driver_points = {}
    for driver in drivers:
        driver_points[driver] = 0
    system = point_systems[system]
    proccess_season_data(qualy_data, drivers, driver_points, system)
