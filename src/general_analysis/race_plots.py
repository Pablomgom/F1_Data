import fastf1

from fastf1 import utils, plotting

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from datetime import timedelta
import math


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
    plt.tight_layout()
    plt.savefig(f"../PNGs/POSITION CHANGES {session.event.OfficialEventName}.png", dpi=400)
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


def driver_laptimes(race):
    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)

    point_finishers = race.drivers[:10]
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


def race_diff(team_1, team_2, session):

    races = []
    session_names = []
    total_laps = []

    for i in range(session):
        session = fastf1.get_session(2023, i + 1, 'R')
        session.load(telemetry=True)
        races.append(session)
        session_names.append(session.event['Location'].split('-')[0])
        total_laps.append(session.total_laps)

    team_1_times = []
    team_2_times = []

    team_1_total_laps = []
    team_2_total_laps = []

    for race in races:
        team_1_laps = race.laps.pick_team(team_1)
        team_1_laps = team_1_laps.dropna(subset='LapTime')
        #team_1_laps = team_1_laps[team_1_laps['PitOutTime'].isna()]
        #team_1_laps = team_1_laps[team_1_laps['PitInTime'].isna()]
        team_1_laps = team_1_laps[team_1_laps['TrackStatus'].astype(str).apply(lambda x: set(x) <= {'1', '2'})]

        if team_1 == 'Alpine' and race.event['Location'] == 'Budapest':
            team_1_times.append(timedelta(hours=0, minutes=0, seconds=0, milliseconds=0))
            team_1_total_laps.append(0)
        else:
            team_1_times.append(team_1_laps['LapTime'].sum())
            team_1_total_laps.append(team_1_laps['LapTime'].count())


        team_2_laps = race.laps.pick_team(team_2)
        team_2_laps = team_2_laps.dropna(subset='LapTime')
        #team_2_laps = team_2_laps[team_2_laps['PitOutTime'].isna()]
        #team_2_laps = team_2_laps[team_2_laps['PitInTime'].isna()]
        team_2_laps = team_2_laps[team_2_laps['TrackStatus'].astype(str).apply(lambda x: set(x) <= {'1', '2'})]

        team_2_times.append(team_2_laps['LapTime'].sum())
        team_2_total_laps.append(team_2_laps['LapTime'].count())

    delta_laps = []

    for i in range(len(team_2_times)):

        if team_1_total_laps[i] == 0:
            delta_laps.append(0)
        else:
            mean_time_team_1 = (team_1_times[i] / team_1_total_laps[i]).total_seconds() * 1000
            mean_time_team_2 = (team_2_times[i] / team_2_total_laps[i]).total_seconds() * 1000

            delta = ((mean_time_team_2 - mean_time_team_1) / mean_time_team_2) * total_laps[i]
            delta_laps.append(delta)

    plt.figure(figsize=(13, 7))
    delta_laps = [x if not math.isnan(x) else 0 for x in delta_laps]

    for i in range(len(session_names)):
        color = plotting.team_color(team_1) if delta_laps[i] > 0 else plotting.team_color(team_2)
        label = f'{team_1} faster' if delta_laps[i] > 0 else f'{team_2} faster'
        plt.bar(session_names[i], delta_laps[i], color=color, label=label)

    # Add exact numbers above or below every bar based on whether it's a maximum or minimum
    for i in range(len(session_names)):
        if delta_laps[i] > 0:  # If the bar is above y=0
            plt.text(session_names[i], delta_laps[i] + 0.06, "{:.2f} %".format(delta_laps[i]),
                     ha='center', va='top')
        else:  # If the bar is below y=0
            plt.text(session_names[i], delta_laps[i] - 0.075, "{:.2f} %".format(delta_laps[i]),
                     ha='center', va='bottom')

    # Set the labels and title
    plt.ylabel(f'Percentage time difference', fontsize=14)
    plt.xlabel('Circuito', fontsize=14)
    plt.title(f'{team_1} VS {team_2} race time difference', fontsize=14)

    step = 0.2

    start = np.ceil(abs(min(delta_laps) / step))
    if min(delta_laps) < 0:
        start = np.floor(min(delta_laps) / step) * step
    else:
        start = np.ceil(min(delta_laps) / step) * step
    end = np.ceil(max(delta_laps) / step) * step

    delta_laps = [x for x in delta_laps if x != 0]
    mean_y = np.mean(delta_laps)
    # Draw horizontal line at y=mean_y
    plt.axhline(mean_y, color='red', linewidth=2, label='Mean distance')
    if min(delta_laps) < 0:
        plt.axhline(0, color='black', linewidth=2)

    # Generate a list of ticks from minimum to maximum y values considering 0.0 value and step=0.2
    yticks = list(np.arange(start, end + step, step))
    yticks.append(mean_y)
    delete = None
    for i in range(len(yticks)):
        if abs(yticks[i] - mean_y) <= 0.10:
            delete = yticks[i]
            break
    if delete is not None:
        yticks.remove(delete)
    yticks = sorted(yticks)

    plt.yticks(yticks, [f'{tick:.2f} %' if tick != mean_y else f'{tick:.2f} %' for tick in yticks])

    # To avoid repeating labels in the legend, we handle them separately
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.savefig(f"../PNGs/{team_2} VS {team_1} race time difference.png", dpi=400)

    # Show the plot
    plt.show()


def race_distance(session, driver_1, driver_2):
    laps_driver_1 = session.laps.pick_driver(driver_1).reset_index()
    laps_driver_2 = session.laps.pick_driver(driver_2).reset_index()

    laps_diff = []
    laps = []
    for i in range(len(laps_driver_1)):
        laps_diff.append(laps_driver_1['LapTime'][i].total_seconds() - laps_driver_2['LapTime'][i].total_seconds())
        laps.append(i+1)

    progressive_sum = np.cumsum(laps_diff)
    colors = ['red']
    for i in range(len(progressive_sum) - 1):
        if progressive_sum[i] < progressive_sum[i + 1]:
            colors.append('green')
        else:
            colors.append('red')
    plt.figure(figsize=(16, 8))
    # Bar Plot
    bars = plt.bar(laps, progressive_sum, color=colors, width=0.9)

    # Annotate bars with their values
    for bar in bars:
        yval = bar.get_height()
        offset = 0.5 if yval > 0 else -0.5  # This will adjust the position above or below the bar. Modify the value if needed.
        plt.annotate(
            f'{yval:.2f}',  # Format to 2 decimal places, modify as needed
            (bar.get_x() + bar.get_width() / 2, yval + offset),  # Adjusted the y-coordinate here
            ha='center',  # horizontal alignment
            va='center',  # vertical alignment
            fontsize=8 # font size
        )
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', edgecolor='green', label='Driver 1 Ahead'),
                       Patch(facecolor='red', edgecolor='red', label='Driver 2 Ahead')]
    plt.legend(handles=legend_elements, loc='best')

    plt.xlabel('Laps', fontsize=20)
    plt.ylabel('Progressive Time Difference (seconds)', fontsize=20)
    plt.title('Progressive Time Difference between Two Drivers', fontsize=20)
    plt.grid(True, axis='y')

    # Display the plot
    plt.tight_layout()  # Adjusts plot parameters for a better layout
    plt.show()




