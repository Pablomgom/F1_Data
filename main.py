import laps
import fastf1
import pandas as pd
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    laps.overlay_race(2022,'Australia','R','MAG','MSC')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    fastf1.Cache.enable_cache('Cache')
    #race = fastf1.get_session(2022,'Australia','R')
    #race.load()

    #print(race.laps)

    #print(race.laps.loc[race.laps['DriverNumber']=='11'])

    #laps.show_race(2022,'Australia','R','PER')

    #laps.show_fastest_lap_qualy(2019,'Monza','LEC')
    laps.show_fastest_lap_qualy(2022,'Monaco','LEC')

    #laps.show_speed_changes(2021,'Monaco')
    #laps.show_speed_changes(2021, 'Monza')

    #laps.compare_two_laps(2021,'Imola','LEC','HAM')


