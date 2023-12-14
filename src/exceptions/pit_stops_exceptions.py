from src.ergast_api.ergast_struct import ergast_struct


def filter_pit_stops(races):

    filtered_races = []
    for r in races.content:

        year = r['year'].iloc[0]
        round = r['round'].iloc[0]

        if year == 2014 and round == 15:
            r = r[r['lap'] != 2]
        elif year == 2015 and round == 10:
            r = r[~((r['fullName'] == 'Will Stevens') & (r['stop'] == 3))]
            r = r[~((r['fullName'] == 'Roberto Merhi') & (r['stop'] == 4))]
            r = r[~r['lap'].isin([44, 45, 46])]
        elif year == 2016 and round == 1:
            r = r[~((r['fullName'] == 'Kevin Magnussen') & (r['lap'] == 17))]
            r = r[r['lap'] != 18]
        elif year == 2016 and round == 13:
            r = r[r['lap'] != 9]
        elif year == 2016 and round == 15:
            r = r[r['lap'] != 1]
        elif year == 2016 and round == 20:
            r = r[~r['lap'].isin([20, 28])]
        elif year == 2017 and round == 2:
            r = r[~r['lap'].isin([4, 5, 6])]
        elif year == 2017 and round == 8:
            r = r[~r['lap'].isin([17, 22])]
            r = r[~r['duration'].isin(['26:23.848', '26:34.609'])]
        elif year == 2017 and round == 14:
            r = r[~r['lap'].isin([1, 2, 3])]
        elif year == 2017 and round == 19:
            r = r[~r['lap'].isin([1, 2, 3])]
        elif year == 2020 and round == 8:
            r = r[r['lap'] != 26]
        elif year == 2020 and round == 9:
            r = r[~r['lap'].isin([7, 8, 45])]
            r = r[~r['duration'].isin(['19:40.725', '19:41.863'])]
        elif year == 2020 and round == 15:
            r = r[r['lap'] != 1]
        elif year == 2021 and round == 2:
            r = r[~r['lap'].isin([32, 33])]
            r = r[~r['duration'].isin(['24:46.154', '25:21.462'])]
        elif year == 2021 and round == 3:
            r = r[~r['lap'].isin([2, 3])]
        elif year == 2021 and round == 6:
            r = r[~r['lap'].isin([46, 47, 48])]
        elif year == 2021 and round == 10:
            r = r[~r['lap'].isin([2])]
        elif year == 2021 and round == 11:
            r = r[~r['lap'].isin([2])]
        elif year == 2021 and round == 19:
            r = r[~r['lap'].isin([6, 7, 8])]
        elif year == 2021 and round == 21:
            r = r[~r['lap'].isin([13, 15])]
        elif year == 2022 and round == 7:
            r = r[~r['lap'].isin([29, 30])]
        elif year == 2022 and round == 10:
            r = r[~r['lap'].isin([1])]
        elif year == 2022 and round == 18:
            r = r[~r['lap'].isin([2])]

        filtered_races.append(r)
    return ergast_struct(filtered_races)
