from datetime import timedelta

import nltk
import pandas as pd

from src.ergast_api.ergast_struct import ergast_struct
from src.exceptions.pit_stops_exceptions import filter_pit_stops
from src.utils.utils import get_country_names


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


def get_list_dataframes(df, schema, pit_stop=False):
    grouped = df.groupby('raceId')
    races_list = [group for _, group in grouped]

    def custom_sort_key(df):
        if pit_stop:
            return df['year'].values[0], df['round'].values[0], df['lap'].values[0]
        else:
            return df['year'].values[0], df['round'].values[0]

    races_list = sorted(races_list, key=custom_sort_key)
    if pit_stop:
        races_list = [apply_custom_schema(df, schema).sort_values(by='lap', ascending=True) for df in races_list]
    else:
        races_list = [apply_custom_schema(df, schema).sort_values(by='position', ascending=True) for df in races_list]
    return races_list


race_results_schema = ['year', 'round', 'raceId', 'raceName', 'driverCurrentNumber', 'grid',
                       'position', 'points', 'laps', 'totalRaceTime',
                       'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed',
                       'constructorRef', 'constructorName', 'constructorNationality',
                       'driverRef', 'driverCode', 'givenName', 'familyName', 'fullName', 'dob',
                       'driverNationality', 'status', 'circuitRef', 'circuitName',
                       'location', 'country']

qualy_results_schema = ['year', 'round', 'raceId', 'number', 'position', 'q1', 'q2', 'q3',
                        'raceName', 'constructorRef', 'constructorName',
                        'givenName', 'familyName', 'fullName', 'driverCode', 'circuitName',
                        'location', 'country', 'circuitRef']

pit_stop_schema = ['year', 'round', 'raceName', 'fullName', 'stop', 'lap', 'pitTime', 'duration']

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
        self.qualifying['position'] = pd.to_numeric(self.qualifying['position'], errors='coerce').fillna(0).astype(int)
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
        results['fullName'] = results['givenName'] + ' ' + results['familyName']
        races_list = get_list_dataframes(results, race_results_schema)
        race_results = ergast_struct(races_list)
        return race_results

    def get_sprint_results(self, year, race_id=None):
        races = self.races
        if race_id is None:
            races = races[races['year'].isin(year)]
        else:
            races = races[(races['year'].isin(year)) & (races['round'] == race_id)]
        results = pd.merge(races, self.sprint_results, on='raceId', how='inner')
        results = pd.merge(results, self.constructors, on='constructorId', how='inner')
        results = pd.merge(results, self.drivers, on='driverId', how='inner')
        results = pd.merge(results, self.status, on='statusId', how='inner')
        results = pd.merge(results, self.circuits, on='circuitId', how='inner')
        results['fastestLapTime'] = results['fastestLapTime'].apply(string_to_timedelta)
        results['totalRaceTime'] = results['totalRaceTime'].apply(string_to_timedelta)
        results['rank'] = None
        results['fastestLapSpeed'] = None
        results['fullName'] = results['givenName'] + ' ' + results['familyName']
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
        results['fullName'] = results['givenName'] + ' ' + results['familyName']
        qualy_list = get_list_dataframes(results, qualy_results_schema)
        qualy_results = ergast_struct(qualy_list)
        return qualy_results

    def insert_qualy_data(self, year, round, data):

        self.qualifying = load_csv('qualifying', False)
        raceId = self.races[(self.races['year'] == year) & (self.races['round'] == round)]['raceId'].values[0]
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        data_lines = data.strip().split("\n")
        processed_data = []
        for line in data_lines:
            columns = line.split("\t")
            country = f'{get_country_names(columns[2])}'
            full_name = columns[2].replace(country, '')
            full_name = full_name.replace('United States ', '')
            constructor_name = columns[3].split("-")[0]
            for i in range(4, len(columns)):
                if columns[i] in ['None', '—']:
                    columns[i] = ''
            if full_name == 'JJ Lehto':
                full_name = 'Jyrki Järvilehto'
            if constructor_name == 'Venturi':
                constructor_name = 'Lambo'
            processed_line = [columns[0], columns[1], full_name, constructor_name, columns[4], columns[5]]
            processed_data.append(processed_line)

        data_to_append = pd.DataFrame(processed_data, columns=["position", "number", "fullName",
                                                               "constructorName", "q1", "q2"])

        og_len = len(data_to_append)
        qualyId = self.qualifying['qualifyId'].max() + 1
        self.drivers['fullName'] = self.drivers['givenName'] + ' ' + self.drivers['familyName']
        data_to_append = pd.merge(data_to_append, self.drivers[['driverId', 'fullName']], on='fullName', how='inner')
        data_to_append = pd.merge(data_to_append, self.constructors[['constructorId', 'constructorName']],
                                  on='constructorName', how='inner')
        data_to_append['qualifyId'] = qualyId + data_to_append.index
        data_to_append['raceId'] = raceId
        q_session = ['q1', 'q2', 'q3']
        for q in q_session:
            if q in data_to_append.columns:
                data_to_append[q] = data_to_append[q].astype(str)
            else:
                data_to_append[q] = None
        print(data_to_append.groupby('constructorName').size())
        data_to_append['Valid'] = True
        data_to_append = data_to_append[['qualifyId', 'raceId', 'driverId', 'constructorId',
                                         'number', 'position', 'q1', 'q2', 'q3', 'Valid']]
        new_len = len(data_to_append)
        combined_data = pd.concat([self.qualifying, data_to_append], ignore_index=True)
        if og_len != new_len:
            raise Exception('BAD PROCESS')
        combined_data.to_csv('../resources/ergast_data/qualifying.csv', index=False)

    def get_race_row(self, qualy_data, driver, team, number, grid):
        resultId = self.results['resultId'].max() + 1
        constructorId = self.constructors[self.constructors['constructorName'] == team]['constructorId'].min()
        self.drivers['fullName'] = self.drivers['givenName'] + ' ' + self.drivers['familyName']
        driverId = self.drivers[self.drivers['fullName'] == driver]['driverId'].min()
        raceId = qualy_data['raceId'].min()
        position_text = 'R'
        points = 0
        laps = 0
        statusId = 81

        print(f"""
            {resultId},{raceId},{driverId},{constructorId},{number},{grid},{grid},"{position_text}",
            {grid},{points},{laps},\\N,\\N,\\N,\\N,\\N,\\N,{statusId}
        """)

    def get_qualy_row(self, race_data, driver, team, number, grid):
        qualyId = self.qualifying['qualifyId'].max() + 1
        constructorId = self.constructors[self.constructors['constructorName'] == team]['constructorId'].min()
        self.drivers['fullName'] = self.drivers['givenName'] + ' ' + self.drivers['familyName']
        driverId = self.drivers[self.drivers['fullName'] == driver]['driverId'].min()
        raceId = race_data['raceId'].min()

        print(f"""
            {qualyId},{raceId},{driverId},{constructorId},{number},{grid},/N,/N,/N
        """)


    def get_pit_stops(self, year, race_id=None):
        pitstops = self.pit_stops
        pitstops = pd.merge(pitstops, self.drivers, on='driverId', how='inner')
        pitstops = pd.merge(pitstops, self.races, on='raceId', how='inner')
        if race_id is None:
            pitstops = pitstops[pitstops['year'].isin(year)]
        else:
            pitstops = pitstops[(pitstops['year'].isin(year)) & (pitstops['round'] == race_id)]
        pitstops['fullName'] = pitstops['givenName'] + ' ' + pitstops['familyName']
        pitstops_list = get_list_dataframes(pitstops, pit_stop_schema, True)
        pitstops_list = ergast_struct(pitstops_list)
        pitstops_list = filter_pit_stops(pitstops_list)
        return pitstops_list

