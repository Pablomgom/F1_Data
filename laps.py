import fastf1
from fastf1 import plotting
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from init import *

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