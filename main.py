import laps
import fastf1
import track
import pandas as pd
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #laps.overlay_race(2022,'Australia','R','MAG','MSC')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    fastf1.Cache.enable_cache('Cache')
    race = fastf1.get_session(2022,'Monaco','R')
    race.load()

    #print(race.laps)

    paradas_22=race.laps[pd.notnull(race.laps['PitOutTime'])]

    paradas_22 = paradas_22[paradas_22['LapNumber']==22]

    print(paradas_22)

    carlos=race.laps[race.laps['DriverNumber']=='55']
    checo=race.laps[race.laps['DriverNumber'] == '11']


    carlos = carlos[carlos['LapNumber'].between(15, 25)]
    checo = checo[checo['LapNumber'].between(15, 25)]

    print(carlos)
    print(checo)
    #track.driver_ahead()

    #laps.show_race(2022,'Australia','R','PER')

    #laps.show_fastest_lap_qualy(2021,'Monza','LEC')
    #laps.show_fastest_lap_qualy(2021,'Abu dhabi','LEC')

    #laps.show_speed_changes(2021,'Monaco')
    #laps.show_speed_changes(2021, 'Monza')

    #laps.compare_two_laps(2021,'Imola','VER','HAM')




