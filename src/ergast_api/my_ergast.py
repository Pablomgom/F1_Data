from datetime import timedelta

import pandas as pd

from src.ergast_api.ergast_struct import ergast_struct


def string_to_timedelta(time_str):
    try:
        time_parts = time_str.replace('+', '').split(':')
        if len(time_parts) == 3:
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds, milliseconds = map(float, time_parts[2].split('.'))
            return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
        elif len(time_parts) == 2:
            minutes = int(time_parts[0])
            seconds, milliseconds = map(float, time_parts[1].split('.'))
            return timedelta(minutes=minutes, seconds=seconds, milliseconds=milliseconds)
        elif len(time_parts) == 1:
            seconds, milliseconds = map(float, time_parts[0].split('.'))
            return timedelta(seconds=seconds, milliseconds=milliseconds)
        else:
            print('Fecha con formato chungo')
    except:
        return pd.NaT


def load_csv(csv, is_qualy=True):

    data = pd.read_csv(f'../resources/ergast_data/{csv}.csv', sep=',')
    if is_qualy:
        for col in data.columns:
            if col in ['q1', 'q2', 'q3']:
                data[col] = data[col].apply(string_to_timedelta)
    return data


def apply_custom_schema(df, schema):
    return df[schema].copy()


def get_list_dataframes(df, schema):
    grouped = df.groupby('raceId')
    races_list = [group for _, group in grouped]

    def custom_sort_key(df):
        # Extract the values from the DataFrame and return them
        return (df['year'].values[0], df['round'].values[0])

    # Sort the list of DataFrames using the custom sorting key
    races_list = sorted(races_list, key=custom_sort_key)
    races_list = [apply_custom_schema(df, schema) for df in races_list]
    return races_list


race_results_schema = ['year', 'raceName', 'driverCurrentNumber', 'grid',
                       'positionOrder', 'points', 'laps', 'totalRaceTime',
                       'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed',
                       'constructorRef', 'constructorName', 'constructorNationality',
                       'driverRef', 'driverCode', 'givenName', 'familyName', 'dob',
                       'driverNationality', 'status', 'circuitRef', 'circuitName',
                       'location', 'country']

qualy_results_schema = ['number', 'position', 'q1', 'q2', 'q3', 'year', 'raceName', 'constructorRef',
                        'constructorName', 'driverCode', 'givenName', 'familyName', 'circuitName',
                        'location', 'country']


class My_Ergast:

    def __init__(self):
        self.circuits = load_csv('circuits')
        self.constructor_results = load_csv('constructor_results')
        self.constructor_standings = load_csv('constructor_standings')
        self.constructors = load_csv('constructors')
        self.driver_standings = load_csv('driver_standings')
        self.drivers = load_csv('drivers')
        self.lap_times = load_csv('lap_times')
        self.pit_stops = load_csv('pit_stops')
        self.qualifying = load_csv('qualifying')
        self.races = load_csv('races')
        self.results = load_csv('results')
        self.seasons = load_csv('seasons')
        self.sprint_results = load_csv('sprint_results')
        self.status = load_csv('status')

    def get_race_results(self, year, race_id=None):
        races = self.races
        if race_id is None:
            races = races[races['year'].isin(year)]
        else:
            races = races[(races['year'].isin(year)) & (races['round'] == race_id)]
        results = pd.merge(races, self.results, on='raceId', how='inner')
        results = pd.merge(results, self.constructors, on='constructorId', how='inner')
        results = pd.merge(results, self.drivers, on='driverId', how='inner')
        results = pd.merge(results, self.status, on='statusId', how='inner')
        results = pd.merge(results, self.circuits, on='circuitId', how='inner')
        results['fastestLapTime'] = results['fastestLapTime'].apply(string_to_timedelta)
        results['totalRaceTime'] = results['totalRaceTime'].apply(string_to_timedelta)
        races_list = get_list_dataframes(results, race_results_schema)
        race_results = ergast_struct(races_list)
        return race_results

    def get_qualy_results(self, year, race_id=None):
        qualys = self.qualifying
        qualys = pd.merge(qualys, self.races, on='raceId', how='inner')
        if race_id is None:
            qualys = qualys[qualys['year'].isin(year)]
        else:
            qualys = qualys[(qualys['year'].isin(year)) & (qualys['round'] == race_id)]
        results = pd.merge(qualys, self.constructors, on='constructorId', how='inner')
        results = pd.merge(results, self.drivers, on='driverId', how='inner')
        results = pd.merge(results, self.circuits, on='circuitId', how='inner')
        results = results.sort_values(by=['year', 'round'], ascending=[True, True])
        qualy_list = get_list_dataframes(results, qualy_results_schema)
        qualy_results = ergast_struct(qualy_list)
        return qualy_results

    def insert_qualy_data(self, year, round):

        self.qualifying = load_csv('qualifying', False)
        new_column_names = {
            'Driver': 'fullName',
            'Pos': 'position',
            'No': 'number',
            'Lap': 'q1',
            'Constructor': 'constructorName',
        }

        teams_dict = {
            'McLaren-Mercedes': 'McLaren',
            'Jordan-Honda': 'Jordan',
            'Williams-BMW': 'Williams',
            'BAR-Honda': 'BAR',
            'Sauber-Petronas': 'Sauber',
            'Jaguar-Cosworth': 'Jaguar',
            'Prost-Acer': 'Prost',
            'Arrows-Asiatech': 'Arrows',
            'Benetton-Renault': 'Benetton',
            'Minardi-European': 'Minardi'
        }

        raceId = self.races[(self.races['year'] == year) & (self.races['round'] == round)]['raceId'].values[0]
        data_to_append = load_csv('take_data', False)
        qualyId = self.qualifying['qualifyId'].max() + 1
        self.drivers['fullName'] = self.drivers['givenName'] + ' ' + self.drivers['familyName']
        data_to_append.rename(columns=new_column_names, inplace=True)
        data_to_append = pd.merge(data_to_append, self.drivers[['driverId', 'fullName']], on='fullName', how='inner')
        data_to_append['constructorName'] = data_to_append['constructorName'].replace(teams_dict)
        data_to_append = pd.merge(data_to_append, self.constructors[['constructorId', 'constructorName']],
                                  on='constructorName', how='inner')
        data_to_append['qualifyId'] = qualyId + data_to_append.index
        data_to_append['raceId'] = raceId
        data_to_append['q2'] = '/N'
        data_to_append['q3'] = '/N'
        data_to_append['q1'] = data_to_append['q1'].astype(str)
        data_to_append['q2'] = data_to_append['q2'].astype(str)
        data_to_append['q3'] = data_to_append['q3'].astype(str)

        data_to_append = data_to_append[['qualifyId', 'raceId', 'driverId', 'constructorId',
                                         'number', 'position', 'q1', 'q2', 'q3']]
        combined_data = pd.concat([self.qualifying, data_to_append], ignore_index=True)
        combined_data.to_csv('../resources/ergast_data/qualifying.csv', index=False)
        a = 1
