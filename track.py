import fastf1
import numpy as np
from fastf1 import plotting
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from init import *
import datetime


def driver_ahead():
    fastf1.plotting.setup_mpl()
    # fastf1.Cache.enable_cache("path/to/cache")

    session = fastf1.get_session(2022, 'Monaco', 'R')
    session.load()

    DRIVER = 'LEC'  # which driver; need to specify number and abbreviation
    DRIVER_NUMBER = '16'
    LAP_N = 22 # which lap number to plot

    drv_laps = session.laps.pick_driver(DRIVER)
    drv_lap = drv_laps[(drv_laps['LapNumber'] == LAP_N)]  # select the lap

    # create a matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot()

    # ############### new
    df_new = drv_lap.get_car_data().add_driver_ahead()
    ax.plot(df_new['Time'], df_new['DistanceToDriverAhead'], label=DRIVER)


    plt.legend()
    plt.show()

def time_distance_in_race(year, gp, driver_1, driver_2):
    plotting.setup_mpl()

    fastf1.Cache.enable_cache('Cache')  # optional but recommended

    race = fastf1.get_session(year, gp, 'R')
    race.load()

    driver_1_race = race.laps.pick_driver(driver_1)
    driver_2_race = race.laps.pick_driver(driver_2)

    driver_1_team = driver_1_race['Team'].unique()[0]
    driver_2_team = driver_2_race['Team'].unique()[0]

    for i in range(len(driver_1_race)):
        if (driver_1_race.iat[i,0]<=driver_2_race.iat[i,0]):
            print(driver_1_race.iat[i,0]-driver_2_race.iat[i,0])
            print(driver_1)
        else:
            print(driver_1_race.iat[i,0]-driver_2_race.iat[i,0])
            print(driver_2)
        print("LAP: " + str(i+1))



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