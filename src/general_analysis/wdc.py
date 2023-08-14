import fastf1
import numpy as np
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt, cm


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




def wdc_comparation(driver, start=None, end=None):

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
        append = False

        for race in races.content:
            race_data = race[race['familyName'] == driver]
            if len(race_data) > 0:
                driver_team = race[race['familyName'] == driver]['constructorName'].values[0]
                team_mates = race[race['constructorName'] == driver_team]
                team_mates = team_mates[team_mates['familyName'] != driver]
                points += race_data['points'].values[0]
                team_mate_points += team_mates['points'].values[0]
                team_mates_names.append(team_mates['driverCode'].values[0])
                append = True

        for race in sprints.content:
            race_data = race[race['familyName'] == driver]
            if len(race_data) > 0:
                driver_team = race[race['familyName'] == driver]['constructorName'].values[0]
                team_mates = race[race['constructorName'] == driver_team]
                team_mates = team_mates[team_mates['familyName'] != driver]
                points += race_data['points'].values[0]
                team_mate_points += team_mates['points'].values[0]
                team_mates_names.append(team_mates['driverCode'].values[0])
                append = True
        if append:
            points_per_year.append(points)
            team_mates_points_per_year.append(team_mate_points)
            if driver == 'Alonso' and i == 2001:
                team_mates_names_per_year.append(['MAR - YOO'])
            else:
                team_mates_names_per_year.append(team_mates_names)
            driver_code.append(races.content[0][races.content[0]['familyName'] == driver]['driverCode'])
            years.append(i)
        print(i)

    fixed_names = []
    for year in team_mates_names_per_year:
        unique_names = set(year)
        formatted_elements = [f"{elem}" for elem in unique_names]
        formatted_string = ' - '.join(formatted_elements)
        fixed_names.append(formatted_string)

    plt.figure(figsize=(20, 8))
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
            edgecolor='white', label=driver_code[0].values[0])

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
            plt.text(x, value + offset, '{:.0f}'.format(value), ha='center', va='bottom', zorder=5
                     )
    add_labels(r1, points_per_year)
    add_labels(r2, team_mates_points_per_year)

    # Add some details
    plt.title('Driver Points by Year')
    plt.xlabel('Year', fontweight='bold')
    plt.ylabel('Points', fontweight='bold')
    plt.xticks([(x1 + x2) / 2 for x1, x2 in zip(r1, r2)], years)

    plt.legend(loc='best')

    plt.title(f'{driver} points comparation per year with his teammates')
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.gca().xaxis.grid(False)
    plt.tick_params(axis='x', which='both', pad=15)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver} POINTS COMPARATION WDC.png', dpi=300)

    plt.show()