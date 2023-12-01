import fastf1
import numpy as np
import matplotlib.path as mpath
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt, cm, image as mpimg
import re
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src.plots.plots import rounded_top_rect, stacked_bars


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

    fig, ax1 = plt.subplots(figsize=(7.2, 6.5), dpi=150)
    # Bar width
    barWidth = 0.4

    # Set position of bars on x axis
    r1 = np.arange(len(points_per_year))
    r2 = [x + barWidth for x in r1]

    all_teammate_names = set()
    for names in fixed_names:
        all_teammate_names.add(names)

    # Using a colormap for generating unique colors
    color_map = cm.get_cmap('tab20', len(all_teammate_names))
    name_to_color = {name: color_map(i) for i, name in enumerate(all_teammate_names)}

    # Create bars for Driver 1
    bars = plt.bar(r1, points_per_year, color="#8B0000", width=barWidth,
                   edgecolor='white')

    def my_rounded_top_rect(x, y, width, height, color):
        corner_radius = min(5 * width, height / 2)

        # Calculate the starting point for the curves based on height
        # Calculate the starting point for the curves based on height
        curve_start_y = y + height * 0.98 - corner_radius
        curve_end_x_left = x + width / 2
        curve_end_x_right = x + width / 2

        # Vertices for the rectangle with rounded top
        verts = [
            (x, y),  # bottom-left
            (x, curve_start_y),  # start of left curve
            (x, y + height),  # Control point for top-left curve
            (curve_end_x_left, y + height),  # end of left curve and start of top-left curve
            (curve_end_x_right, y + height),  # end of top-left curve and start of top-right curve
            (x + width, y + height),  # Control point for top-right curve
            (x + width, curve_start_y),  # end of top-right curve
            (x + width, y),  # bottom-right
            (x, y)  # close polygon
        ]

        codes = [
            mpath.Path.MOVETO,
            mpath.Path.LINETO,
            mpath.Path.CURVE3,
            mpath.Path.CURVE3,
            mpath.Path.LINETO,
            mpath.Path.CURVE3,
            mpath.Path.CURVE3,
            mpath.Path.LINETO,
            mpath.Path.CLOSEPOLY
        ]

        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor=color, edgecolor=color)
        return patch

    for bar in bars:
        bar.set_alpha(0)

    # Overlay rounded rectangle patches on top of the original bars
    for bar in bars:
        height = bar.get_height()
        x, y = bar.get_xy()
        width = bar.get_width()
        rounded_box = my_rounded_top_rect(x, y, width, height, "#8B0000")
        rounded_box.set_facecolor("#8B0000")
        ax1.add_patch(rounded_box)

    added_to_legend = set()
    # Create bars for Driver 2 with varying names
    for i, (r, point, name) in enumerate(zip(r2, team_mates_points_per_year, fixed_names)):
        # Use the appropriate color from our map
        color = name_to_color[name]

        # Create the original bar for the legend
        if name not in added_to_legend:
            bar = plt.bar(r, point, width=barWidth, edgecolor='white', color=color)
            added_to_legend.add(name)
        else:
            bar = plt.bar(r, point, width=barWidth, edgecolor='white', color=color)

        # Hide the original bar
        for b in bar:
            b.set_alpha(0)

        # Create rounded rectangle patch and add it to the axes
        height = bar[0].get_height()
        x, y = bar[0].get_xy()
        width = bar[0].get_width()
        rounded_box = my_rounded_top_rect(x, y, width, height, color)
        ax1.add_patch(rounded_box)

    def add_labels(x_positions, values, offset=0.5):
        for x, value in zip(x_positions, values):
            plt.text(x, value + offset, '{:.1f}'.format(value), ha='center', va='bottom', zorder=5,
                     font='Fira Sans', fontsize=8)

    add_labels(r1, points_per_year)
    add_labels(r2, team_mates_points_per_year)

    legend_lines = [Line2D([0], [0], color="#8B0000", lw=4)]
    names_legend = [driver.split(" ")[1]]
    colors_added = []
    for name in fixed_names:
        color = name_to_color[name]
        if color not in colors_added:
            hex_code = "#{:02X}{:02X}{:02X}".format(round(255 * color[0]),
                                                    round(255 * color[1]), round(255 * color[2]))
            line = Line2D([0], [0], color=hex_code, lw=4)
            legend_lines.append(line)
            names_legend.append(name)
            colors_added.append(color)

    plt.legend(legend_lines, names_legend,
               loc='upper left', fontsize='medium')
    # Add some details

    plt.xlabel('Year', font='Fira Sans', fontsize=11)
    plt.ylabel('Points', font='Fira Sans', fontsize=11)
    plt.xticks([(x1 + x2) / 2 for x1, x2 in zip(r1, r2)], years)

    plt.title(f'{driver} points comparison per year with his teammates {"- excluding mechanical DNFs" if DNFs else ""}',
              font='Fira Sans', fontsize=14,
              )
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.gca().xaxis.grid(False)
    plt.tick_params(axis='x', which='both', pad=15)
    plt.xticks(font='Fira Sans', fontsize=9.5)  # for x-axis
    plt.yticks(font='Fira Sans', fontsize=9.5)  # for y-axis
    plt.figtext(0.01, 0.02, '@Big_Data_Master', font='Fira Sans', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver} POINTS COMPARISON WDC {"DNFs" if DNFs is True else ""}.png', dpi=150)

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
