from fastf1 import utils, plotting
from matplotlib import pyplot as plt, cm
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


def position_changes(session):

    plotting.setup_mpl(misc_mpl_mods=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.9))

    for drv in session.drivers:
        drv_laps = session.laps.pick_driver(drv)

        abb = drv_laps['Driver'].iloc[0]
        color = plotting.driver_color(abb)

        starting_grid = session.results.GridPosition.to_frame().reset_index(drop=False).rename(columns={'index':'driver'})
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
    plt.savefig("../PNGs/VER-PER QUALY LAPS SPA.png", dpi=400)
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
    plt.savefig("../PNGs/NOR LAPS SPA.png", dpi=400)
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
            patches[label] = mpatches.Patch(color=color, label=label)

            previous_stint_end += row["StintLength"]

    plt.legend(handles=patches.values(), bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=2)

    fig.suptitle(session.event.OfficialEventName, fontsize=12)
    plt.xlabel("Lap Number")
    plt.grid(False)
    # invert the y-axis so drivers that finish higher are closer to the top
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    plt.savefig("../PNGs/TYRE STRATEGY SPA.png", dpi=400)

    plt.show()
