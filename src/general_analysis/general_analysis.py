import fastf1
import pandas as pd
from fastf1 import utils, plotting, events
from fastf1.core import Laps
from matplotlib import pyplot as plt, cm
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from timple.timedelta import strftimedelta


def position_changes(session):
    plotting.setup_mpl(misc_mpl_mods=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.9))

    for drv in session.drivers:
        drv_laps = session.laps.pick_driver(drv)

        abb = drv_laps['Driver'].iloc[0]
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
    plt.tight_layout()
    plt.savefig(f"../PNGs/POSITION CHANGES {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def overlying_laps(session, driver_1, driver_2):
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'

    # Set the color of text, labels, and ticks to white
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    d1_lap = session.laps.pick_driver(driver_1).pick_fastest()
    d2_lap = session.laps.pick_driver(driver_2).pick_fastest()

    delta_time, ref_tel, compare_tel = utils.delta_time(d1_lap, d2_lap)
    # ham is reference, lec is compared

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the speed for driver 1 (reference) and driver 2 (compare)
    ax.plot(ref_tel['Distance'], ref_tel['Speed'],
            color='#0000FF',
            label=driver_1)
    ax.plot(compare_tel['Distance'], compare_tel['Speed'],
            color='#FFA500',
            label=driver_2)

    # Plot the delta time on a secondary y-axis (twinx)
    twin = ax.twinx()
    twin.plot(ref_tel['Distance'], delta_time, '--', color='white', alpha=0.5, label='delta')
    twin.set_ylabel(f"<-- {driver_2} ahead | {driver_1} ahead -->")

    # Set the labels for the axes
    ax.set_xlabel('Distance')
    ax.set_ylabel('Speed')
    ax.set_title(f'Qualy lap comparation between {driver_1} and {driver_2}')

    # Show the legend

    # Get the legend handles and labels from the first axes
    handles1, labels1 = ax.get_legend_handles_labels()

    # Get the legend handles and labels from the second (twin) axes
    handles2, labels2 = twin.get_legend_handles_labels()

    # Combine the handles and labels
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Create the legend
    ax.legend(handles, labels, loc='lower right', bbox_to_anchor=(0, -0.1))

    # Display the plot
    plt.savefig(f"../PNGs/{driver_1} - {driver_2} QUALY LAPS {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def driver_lap_times(race, driver):
    plotting.setup_mpl(misc_mpl_mods=False)

    driver_laps = race.laps.pick_driver(driver).pick_quicklaps().reset_index()

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(data=driver_laps,
                    x="LapNumber",
                    y="LapTime",
                    ax=ax,
                    hue="Compound",
                    palette=plotting.COMPOUND_COLORS,
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
    plt.savefig(f"../PNGs/{driver} LAPS {race.event.OfficialEventName}.png", dpi=400)
    plt.show()


def gear_changes(session, driver):
    lap = session.laps.pick_driver(driver).pick_fastest()
    tel = lap.get_telemetry()

    x = np.array(tel['X'].values)
    y = np.array(tel['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    gear = tel['nGear'].to_numpy().astype(float)

    cmap = cm.get_cmap('Paired')
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N + 1), cmap=cmap)
    lc_comp.set_array(gear)
    lc_comp.set_linewidth(4)

    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    title = plt.suptitle(
        f"Fastest Lap Gear Shift Visualization\n"
        f"{lap['Driver']} - {session.event['EventName']} {session.event.year}"
    )

    cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
    cbar.set_ticks(np.arange(1.5, 9.5))
    cbar.set_ticklabels(np.arange(1, 9))
    plt.savefig(f"../PNGs/GEAR CHANGES {driver} {session.event.OfficialEventName}", dpi=400)
    plt.show()


def tyre_strategies(session):
    laps = session.laps
    drivers = session.drivers

    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

    stints = laps[["Driver", "Stint", "Compound", "LapNumber", "FreshTyre"]]
    stints = stints.groupby(["Driver", "Stint", "Compound", "FreshTyre"])
    stints = stints.count().reset_index()

    stints = stints.rename(columns={"LapNumber": "StintLength"})

    fig, ax = plt.subplots(figsize=(5, 10))

    patches = {}

    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]

        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            # each row contains the compound name and stint length
            # we can use these information to draw horizontal bars

            if row['FreshTyre']:
                alpha = 1
                color = plotting.COMPOUND_COLORS[row["Compound"]]
            else:
                alpha = 0.5
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


def qualy_results(session):
    drivers = pd.unique(session.laps['Driver'])
    list_fastest_laps = list()
    for drv in drivers:
        drvs_fastest_lap = session.laps.pick_driver(drv).pick_fastest()
        list_fastest_laps.append(drvs_fastest_lap)
    fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

    pole_lap = fastest_laps.pick_fastest()
    fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']
    fastest_laps.dropna(how='all', inplace=True)

    team_colors = list()
    for index, lap in fastest_laps.iterlaps():
        color = plotting.team_color(lap['Team'])
        team_colors.append(color)

    fig, ax = plt.subplots()
    ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'],
            color=team_colors, edgecolor='grey')

    ax.set_yticks(fastest_laps.index)
    ax.set_yticklabels(fastest_laps['Driver'])

    # show fastest at the top
    ax.invert_yaxis()

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

    lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')

    plt.suptitle(f"{session.event['EventName']} {session.event.year} Qualifying\n"
                 f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})")

    plt.xlabel("Seconds")
    plt.ylabel("Driver")

    def custom_formatter(x, pos):
        return x/1e9

    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    plt.savefig(f"../PNGs/QUALY OVERVIEW {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def win_wdc(standings):
    driver_standings = standings.content[0]

    POINTS_FOR_SPRINT = 8 + 25 + 1  # Winning the sprint, race and fastest lap
    POINTS_FOR_CONVENTIONAL = 25 + 1  # Winning the race and fastest lap

    events = fastf1.events.get_events_remaining(force_ergast=True)
    # Count how many sprints and conventional races are left
    sprint_events = len(events.loc[events["EventFormat"] == "sprint"])
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
    plt.title(f'Who can still win the WDC? - Season {standings.description.season[0]} with {len(events)} races remaining')
    plt.legend()

    plt.savefig(f"../PNGs/CAN WIN WDC - SEASON{standings.description.season[0]} AT {len(events)} REMAINING.png", dpi=400)
    # Display the plot
    plt.show()