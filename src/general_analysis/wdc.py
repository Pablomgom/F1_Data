import fastf1
import numpy as np
from PIL import Image
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt, cm, image as mpimg
import re

from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def win_wdc(standings):
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

    plt.figure(figsize=(10, 6))
    plt.bar(driver_standings['driverCode'], driver_standings['points'] + points, label='Maximum Points Possible')
    plt.bar(driver_standings['driverCode'], driver_standings['points'], label='Current Points')
    plt.axhline(y=driver_standings['points'].max(), color='r', linestyle='--', label='Points cap')

    plt.xlabel('Points')
    plt.ylabel('Drivers')
    plt.title(
        f'Who can still win the WDC? - Season {standings.description.season[0]} with {len(events)} races remaining')
    plt.legend()

    plt.savefig(f"../PNGs/CAN WIN WDC - SEASON{standings.description.season[0]} AT {len(events)} REMAINING.png",
                dpi=400)
    # Display the plot
    plt.show()


def process_race_data(content, driver, points, team_mate_points, team_mates_names, DNFs = None):
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

def wdc_comparation(driver, start=None, end=None, DNFs = None):

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

        if i == 1974:
            a = 1

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

    plt.figure(figsize=(25, 8))
    # Bar width
    barWidth = 0.45

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
    plt.bar(r1, points_per_year, width=barWidth, color="#8B0000",
            edgecolor='white', label=driver.split(" ")[0])

    added_to_legend = set()
    # Create bars for Driver 2 with varying names
    for i, (r, point, name) in enumerate(zip(r2, team_mates_points_per_year, fixed_names)):
        # Use the appropriate color from our map
        color = name_to_color[name]
        if name not in added_to_legend:
            plt.bar(r, point, width=barWidth, edgecolor='white', color=color, label=name)
            added_to_legend.add(name)
        else:
            plt.bar(r, point, width=barWidth, edgecolor='white', color=color)

    def add_labels(x_positions, values, offset=0.5):
        for x, value in zip(x_positions, values):
            plt.text(x, value + offset, '{:.1f}'.format(value), ha='center', va='bottom', zorder=5,
                     fontsize=14)
    add_labels(r1, points_per_year)
    add_labels(r2, team_mates_points_per_year)

    # Add some details
    plt.title('Driver Points by Year')
    plt.xlabel('Year', fontweight='bold', fontsize=15)
    plt.ylabel('Points', fontweight='bold', fontsize=15)
    plt.xticks([(x1 + x2) / 2 for x1, x2 in zip(r1, r2)], years)

    plt.legend(loc='best', fontsize=13)

    plt.title(f'{driver} points comparison per year with his teammates {"- excluding mechanical DNFs" if DNFs else ""}',
              fontsize=25
              )
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.gca().xaxis.grid(False)
    plt.tick_params(axis='x', which='both', pad=15)
    plt.xticks(fontsize=14)  # for x-axis
    plt.yticks(fontsize=14)  # for y-axis
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver} POINTS COMPARISON WDC {"DNFs" if DNFs is True else ""}.png', dpi=300)

    plt.show()



def team_wdc_history(team):
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)
    ergast = Ergast()
    years = []
    positions = []
    for i in range(1950, 2024):
        standings = ergast.get_constructor_standings(season=i, limit=1000)
        if len(standings.content) > 0:
            team_data = standings.content[0][standings.content[0]['constructorName'] == team]
            if len(team_data) > 0:
                position = team_data['position'].min()
                years.append(i)
                positions.append(position)

        print(years, positions)

    fig, ax = plt.subplots(figsize=(30, 10))
    plt.plot(years, positions, color='red', marker='o', linewidth=3, zorder=2)

    plt.yticks(np.arange(1, max(positions)+1, 1))
    plt.xticks(np.arange(min(years), max(years) + 1, 4))
    plt.gca().invert_yaxis()

    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    start_year = None
    for i in range(len(positions)):
        # Mark the start of a period where position becomes 1
        if positions[i] == 1 and (i == 0 or positions[i - 1] != 1):
            start_year = years[i]
        # Mark the end of a period where position changes from 1 to something else
        elif positions[i] != 1 and (i > 0 and positions[i - 1] == 1):
            end_year = years[i - 1]
            if start_year == end_year:
                plt.annotate(f'{start_year}', (end_year, 1), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=14)
            else:
                plt.annotate(f'{start_year}-{end_year}', (start_year + ((end_year-start_year)/2), 1),
                             textcoords="offset points", xytext=(0, 10), fontsize=14,
                             ha='center')
            start_year = None  # Reset the start_year
    # Label axes
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Constructors position', fontsize=20)
    plt.title(f'{team.upper()} CONSTRUCTOR POSITIONS HISTORY', fontsize=30, y=1.05)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Load and display the image
    image = mpimg.imread('../resources/Ferrari logo.png')
    imagebox = OffsetImage(image, zoom=0.2)  # You can change the zoom level
    ab = AnnotationBbox(imagebox, (1990, 7), frameon=False)  # Position to display image

    ax.add_artist(ab)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{team} history.png', dpi=400)
    # Display the plot
    plt.show()