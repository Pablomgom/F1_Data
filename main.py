import laps
import fastf1
from fastf1 import plotting
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from init import *
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #laps.overlay_race(2022,'Monaco','R','MAG','MSC')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    fastf1.Cache.enable_cache('Cache')
    race = fastf1.get_session(2022,'Monaco','R')
    race.load()

    #print(race.laps)

    print(race.laps.loc[race.laps['DriverNumber']=='11'])



