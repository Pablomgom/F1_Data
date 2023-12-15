import os

import pandas as pd
import sqlite3


class Database:

    def __init__(self):
        self.db_file_path = '../resources/database.db'
        self.csv_folder = '../resources/csv'
        self.ergast_folder = '../resources/ergast_data'


    def auto_incremental_id(self):
        csv_files = os.listdir(self.csv_folder)
        paths = [os.path.join(self.csv_folder, csv) for csv in csv_files]
        for p in paths:
            file = pd.read_csv(p)
            index = file.index + 1
            file.insert(0, 'ID', index)
            file.to_csv(p, index=False)