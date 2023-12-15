import os

import pandas as pd
import sqlite3


class Database:

    def __init__(self):
        self.db_file_path = 'db/database.db'
        self.csv_folder = '../resources/csv'
        self.ergast_folder = '../resources/ergast_data'


    def auto_incremental_id_my_data(self):
        csv_files = os.listdir(self.csv_folder)
        paths = [os.path.join(self.csv_folder, csv) for csv in csv_files]
        for p in paths:
            file = pd.read_csv(p)
            if 'ID' not in file.columns:
                index = file.index + 1
                file.insert(0, 'ID', index)
                file.to_csv(p, index=False)

    def auto_incremental_id_my_ergast(self):
        csv_files = os.listdir(self.ergast_folder)
        csv_files = [csv for csv in csv_files if csv in ['qualifying.csv', 'results.csv']]
        paths = [os.path.join(self.ergast_folder, csv) for csv in csv_files]
        for p in paths:
            file = pd.read_csv(p)
            if 'qualifyId' in file.columns:
                col_name = 'qualifyId'
            else:
                col_name = 'resultId'
            file = file.drop(col_name, axis=1)
            index = file.index + 1
            file.insert(0, col_name, index)
            file['position'] = file['position'].astype(int)
            file = file.sort_values(by=['raceId', 'position'])
            file.to_csv(p, index=False)
