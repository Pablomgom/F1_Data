from fastf1 import utils, plotting
from matplotlib import pyplot as plt


def position_changes(session):

    plotting.setup_mpl(misc_mpl_mods=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.9))

    for drv in session.drivers:
        drv_laps = session.laps.pick_driver(drv)

        abb = drv_laps['Driver'].iloc[0]
        color = plotting.driver_color(abb)

        ax.plot(drv_laps['LapNumber'], drv_laps['Position'],
                label=abb, color=color)

    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlabel('Lap')
    ax.set_ylabel('Position')

    ax.legend(bbox_to_anchor=(1.0, 1.02))
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
    twin.plot(ref_tel['Distance'], delta_time, '--', color='white', alpha=0.5)
    twin.set_ylabel(f"<-- {driver_2} ahead | {driver_1} ahead -->")

    # Set the labels for the axes
    ax.set_xlabel('Distance')
    ax.set_ylabel('Speed')
    ax.set_title('Sample Plot')

    # Show the legend
    ax.legend(loc='lower right', bbox_to_anchor=(0, -0.1))

    # Display the plot
    plt.show()
