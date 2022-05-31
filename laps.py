import fastf1
from fastf1 import plotting
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from init import *
from matplotlib.collections import LineCollection
from matplotlib import cm
import numpy as np

def overlay_race(year, gp, session, driver_1, driver_2):
    plotting.setup_mpl()

    fastf1.Cache.enable_cache('Cache')  # optional but recommended

    race = fastf1.get_session(year, gp, session)
    race.load()

    driver_1_race = race.laps.pick_driver(driver_1)
    driver_2_race = race.laps.pick_driver(driver_2)

    driver_1_team = driver_1_race['Team'].unique()[0]
    driver_2_team = driver_2_race['Team'].unique()[0]

    print(race.laps['Driver'].unique())

    fig, ax = plt.subplots()
    ax.plot(driver_1_race['LapNumber'], driver_1_race['LapTime'], color=color_dict.get(driver_1_team))

    driver_1_patch = mpatches.Patch(color=color_dict.get(driver_1_team), label=driver_1)
    if (driver_1_team != driver_2_team):
        ax.plot(driver_2_race['LapNumber'], driver_2_race['LapTime'], color=color_dict.get(driver_2_team))
    elif (driver_1_team != 'Haas F1 Team'):
        ax.plot(driver_2_race['LapNumber'], driver_2_race['LapTime'], color='#FFFFFF')
        driver_2_patch = mpatches.Patch(color='blue', label=driver_2)
    else:
        ax.plot(driver_2_race['LapNumber'], driver_2_race['LapTime'], color='#FFFF00')
        driver_2_patch = mpatches.Patch(color='#FFFF00', label=driver_2)

    plt.legend(handles=[driver_1_patch,driver_2_patch])
    ax.set_title(driver_1+" vs "+driver_2)
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")
    plt.show()

def show_race(year, gp, session, driver_1):
    plotting.setup_mpl()

    fastf1.Cache.enable_cache('Cache')  # optional but recommended

    race = fastf1.get_session(year, gp, session)
    race.load()

    driver_1_race = race.laps.pick_driver(driver_1)

    driver_1_team = driver_1_race['Team'].unique()[0]

    print(race.laps['Driver'].unique())

    fig, ax = plt.subplots()
    ax.plot(driver_1_race['LapNumber'], driver_1_race['LapTime'], color=color_dict.get(driver_1_team))

    driver_1_patch = mpatches.Patch(color=color_dict.get(driver_1_team), label=driver_1)

    plt.legend(handles=[driver_1_patch])
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Lap Time')
    ax.set_title(driver_1)

    plt.show()


def show_fastest_lap_qualy(year, gp, driver_1):
    fastf1.plotting.setup_mpl()

    session = fastf1.get_session(year, gp, 'Q')

    session.load()
    fast_leclerc = session.laps.pick_driver(driver_1).pick_fastest()
    lec_car_data = fast_leclerc.get_car_data()
    t = lec_car_data['Time']
    vCar = lec_car_data['Speed']

    # The rest is just plotting
    fig, ax = plt.subplots()
    ax.plot(t, vCar, label='Fast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Speed [Km/h]')
    ax.set_title(driver_1)
    ax.legend()
    plt.show()

def show_speed_changes(year,gp):
    fastf1.Cache.enable_cache('Cache')

    session = fastf1.get_session(year, gp, 'Q')
    session.load()

    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()

    x = np.array(tel['X'].values)
    y = np.array(tel['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    speed = tel['Speed'].to_numpy().astype(float)

    cmap = cm.get_cmap('Paired')
    lc_comp = LineCollection(segments, norm=plt.Normalize(0, cmap.N + int(max(speed))+30), cmap='jet')
    lc_comp.set_array(speed)
    lc_comp.set_linewidth(4)


    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    title = plt.suptitle(
        f"Fastest Lap Visualization\n"
        f"{lap['Driver']} - {session.event['EventName']} {session.event.year}"
    )

    cbar = plt.colorbar(mappable=lc_comp, label="Speed", boundaries=np.arange(0, int(max(speed))+30,30))
    cbar.set_ticks(np.arange(0, int(max(speed))+30,30))
    cbar.set_ticklabels(np.arange(0, int(max(speed))+30,30))

    plt.show()