import os

import pandas as pd
import sqlite3


class Database:

    def __init__(self):
        self.db_file_path = 'db/database.db'
        self.csv_folder = '../resources/csv'
        self.ergast_folder = '../resources/ergast_data'

