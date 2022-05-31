import fastf1
import fastf1.plotting
import fastf1.legacy
import numpy as np
import matplotlib.pyplot as plt

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