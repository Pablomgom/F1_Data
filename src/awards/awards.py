import pickle

import fastf1
import pandas as pd


def get_all_sessions():
    sessions = []
    for i in range(0, 22):
        for j in range(0, 5):
            try:
                session = fastf1.get_session(2023, i + 1, j + 1)
                session.load()
                sessions.append(session)
            except:
                print(f'No data for {i+1}-{j+1}')
    return sessions


def laps(sessions):
    tyres_dict = {}

    for s in sessions:
        drivers = s.laps['Driver'].unique()
        for d in drivers:
            d_laps = s.laps.pick_driver(d)['Compound'].value_counts()
            tyres = d_laps.index.values
            for t in tyres:
                if t not in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET', 'UNKNOWN', 'TEST_UNKNOWN']:
                    if d == 'PER':
                        a = 1
                    print(f'{d} in {s.api_path}')
                    index = \
                        s.laps.pick_driver(d)[s.laps.pick_driver(d)['Compound'] == 'nan'].reset_index().index.values[0]
                    try:
                        t = s.laps.pick_driver(d)['Compound'].reset_index(drop=True).iloc[index + 1]
                    except IndexError:
                        t = s.laps.pick_driver(d)['Compound'].reset_index(drop=True).iloc[index - 1]
                laps = d_laps[t]
                if t == 'TEST_UNKNOWN':
                    t = 'UNKNOWN'
                if d not in tyres_dict:
                    tyres_dict[d] = {t: laps}
                elif d in tyres_dict and t not in list(tyres_dict[d].keys()):
                    tyres_dict[d][t] = laps
                else:
                    previous_laps = tyres_dict[d].get(t)
                    tyres_dict[d][t] = laps + previous_laps

    max_values = {}
    total_laps = {}

    for driver, compounds in tyres_dict.items():
        laps_count = 0
        for compound, n_laps in compounds.items():
            laps_count += n_laps
            if compound not in max_values:
                max_values[compound] = {'drivers': [driver], 'laps': n_laps}
            else:
                if n_laps > max_values[compound]['laps']:
                    max_values[compound] = {'drivers': [driver], 'laps': n_laps}
                elif n_laps == max_values[compound]['laps']:
                    max_values[compound]['drivers'].append(driver)
        total_laps[driver] = laps_count

    for compound, data in max_values.items():
        print(f'{compound} - {data}')

    max_laps = max(total_laps, key=total_laps.get)
    driver = total_laps[max_laps]

    print(f'MAX LAPS: {driver} - {max_laps}')


def longest_stint(sessions):
    stints = {}
    longest_stints_per_tyre = {'SOFT': 0, 'MEDIUM': 0, 'HARD': 0, 'INTERMEDIATE': 0, 'WET': 0}
    longest_stints_details = {'SOFT': {}, 'MEDIUM': {}, 'HARD': {}, 'INTERMEDIATE': {}, 'WET': {}}

    for s in sessions:
        if s.name == 'Race' or s.name == 'Sprint':
            drivers = s.laps['Driver'].unique()
            for d in drivers:
                if s.event.Country == 'Canada' and d == 'TSU':
                    print('TSUNODA THINGS IN CANADA')
                else:
                    d_laps = s.laps.pick_driver(d)
                    s_data = d_laps.groupby(['Compound', 'Stint']).size()
                    tyres = d_laps['Compound'].unique()
                    for t in tyres:
                        t_laps = max(s_data[t].values)
                        if d not in stints:
                            stints[d] = {f'{t}': {'LAPS': t_laps, 'SESSION': s}}
                        elif d in stints and t in stints[d]:
                            if stints[d][t]['LAPS'] < t_laps:
                                stints[d][t] = {'LAPS': t_laps, 'SESSION': s}
                            elif stints[d][t]['LAPS'] == t_laps:
                                stints[d][t]['SESSION'] = f'{stints[d][t]["SESSION"]} - {s}'
                        elif d in stints and t not in stints[d]:
                            stints[d][t] = {'LAPS': t_laps, 'SESSION': s}

                        if t_laps > longest_stints_per_tyre[t]:
                            longest_stints_per_tyre[t] = t_laps
                            longest_stints_details[t] = {d: {'LAPS': t_laps, 'SESSION': s}}
                        elif t_laps == longest_stints_per_tyre[t]:
                            if d not in longest_stints_details[t]:
                                longest_stints_details[t][d] = {'LAPS': t_laps, 'SESSION': s}
                            else:
                                longest_stints_details[t][d][
                                    'SESSION'] = f"{longest_stints_details[t][d]['SESSION']} - {s}"
    print('LONGEST STINTS')
    print(longest_stints_per_tyre, longest_stints_details)


def gear_changes_max_RPM(sessions):
    g_changes = {}
    max_rpm = {}
    top_speed = {}
    for s in sessions:
        drivers = s.laps['Driver'].unique()
        print(f'{s}')
        for d in drivers:
            try:
                d_laps = s.laps.pick_driver(d).get_telemetry()
                changes = d_laps['nGear'].diff().ne(0).sum()
                rpm = max(d_laps['RPM'])
                speed = max(d_laps['Speed'])
                if d in g_changes:
                    g_changes[d] = g_changes[d] + changes
                else:
                    g_changes[d] = changes
                if d not in max_rpm or (d in max_rpm and max_rpm[d]['rpm'] < rpm):
                    max_rpm[d] = {'rpm': rpm, 'session': str(s)}
                if d not in top_speed or (d in top_speed and top_speed[d]['speed'] < speed):
                    top_speed[d] = {'speed': speed, 'session': str(s)}
            except KeyError:
                print(f'No telemetry for {s}')

    g_changes = dict(sorted(g_changes.items(), key=lambda item: item[1], reverse=True))
    max_rpm = dict(sorted(max_rpm.items(), key=lambda item: item[1]['rpm'], reverse=True))
    top_speed = dict(sorted(top_speed.items(), key=lambda item: item[1]['speed'], reverse=True))
    print('GEAR CHANGES')
    print(g_changes)
    print('MAX RPM')
    print(max_rpm)
    print('MAX SPEED')
    print(top_speed)


def pit_stops():
    pit = pd.read_csv('../resources/Pit stops.csv', sep='|')
    pit = pit[pit['Year'] == 2023]
    more_time = pit.groupby('Driver')['Time'].sum().sort_values(ascending=False)
    more_stops = pit['Driver'].value_counts().sort_values(ascending=False)
    print('MORE TIME IN THE PITS')
    print(more_time)
    print('MORE STOPS')
    print(more_stops)


def off_tracks(sessions):
    counter = {}
    spins = {}
    for s in sessions:
        race_control = s.race_control_messages['Message']
        off = list(race_control[race_control.str.contains('OFF TRACK AND CONTINUED')].values)
        deleted = list(race_control[race_control.str.contains('TRACK LIMITS')].values)

        def process_data(messages, data):
            for m in messages:
                driver = m.split(' ')[2].replace('(', '').replace(')', '')
                if m.split(' ')[0] == 'CAR':
                    if driver not in data:
                        data[driver] = 1
                    else:
                        data[driver] = data[driver] + 1
            return data

        counter = process_data(off + deleted, counter)
        spun = list(race_control[race_control.str.contains('SPUN')].values)
        spins = process_data(spun, spins)

    counter = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
    spins = dict(sorted(spins.items(), key=lambda item: item[1], reverse=True))
    print('MORE OFF TRACKS')
    print(counter)
    print('MORE SPINTS')
    print(spins)


def slowest_lap(sessions):
    lap_dict = {}
    for s in sessions:
        print(s)
        drivers = s.laps['Driver'].unique()
        for d in drivers:
            d_laps = s.laps.pick_driver(d)
            for lap in d_laps.iterlaps():
                if pd.isna(lap[1]['PitOutTime']) and pd.isna(lap[1]['PitInTime']):
                    if not pd.isna(lap[1]['LapTime']):
                        stat = str(lap[1].TrackStatus)
                        if '3' not in stat and '4' not in stat and '5' not in stat and '6' not in stat and '7' not in stat:
                            try:
                                avg_speed = lap[1].telemetry['Speed'].mean()
                                if d not in lap_dict or (d in lap_dict and lap_dict[d]['SPEED'] > avg_speed):
                                    lap_dict[d] = {'SPEED': avg_speed, 'SESSION': s}
                            except KeyError:
                                print('NO TEL')
                            except ValueError:
                                print('0 TEL LENGHT')

    lap_dict = dict(sorted(lap_dict.items(), key=lambda item: item[1]['SPEED'], reverse=False))
    print('SLOWEST LAP')
    print(lap_dict)


def race_pace_deviation(sessions):
    dev = {}
    for s in sessions:
        if s.name == 'Race':
            drivers = s.results[s.results['Status'].astype(str).str.contains(r'Finished|\+', regex=True)]['Abbreviation'].values
            for d in drivers:
                d_laps = s.laps.pick_driver(d).pick_wo_box()
                d_laps = d_laps[~d_laps['TrackStatus'].astype(str).str.contains('[34567]')]
                deviation = d_laps['LapTime'].std().total_seconds()
                if d not in dev or (d in dev and dev[d]['DEVIATION'] > deviation):
                    dev[d] = {'DEVIATION': deviation, 'SESSION': s}

    dev = dict(sorted(dev.items(), key=lambda item: item[1]['DEVIATION'], reverse=False))
    print('RACE PACE DEVIATION')
    for k, v in dev.items():
        print(f'{k} - {v}')


def times_lapped(sessions):
    lapped_count = {}
    for s in sessions:
        if s.name == 'Race' or s.name == 'Sprint':
            laps_leader = (s.laps[s.laps['Position'] == 1]['LapStartDate'].drop_duplicates()
                           .to_frame().sort_values(by='LapStartDate', ascending=True).reset_index(drop=True))
            drivers = s.laps['Driver'].unique()
            for d in drivers:
                d_laps = (s.laps.pick_driver(d)['LapStartDate']
                          .to_frame().sort_values(by='LapStartDate', ascending=True).reset_index(drop=True))
                d_laps['Leader'] = -1
                last_lap = d_laps['LapStartDate'].max()
                indexes = laps_leader[laps_leader['LapStartDate'] < last_lap].index.values
                if len(indexes) > 0:
                    reference_laps = laps_leader['LapStartDate'].iloc[0:max(indexes) + 2].to_frame()
                    reference_laps['Leader'] = 1
                    compare_laps = (reference_laps._append(d_laps, ignore_index=True)
                                    .sort_values(by=['LapStartDate', 'Leader'], ascending=[True, False]))
                    compare_laps['Leader_Diff'] = compare_laps['Leader'].cumsum()
                    l_lapped = (compare_laps['Leader_Diff'] >= 2).sum()
                    if l_lapped == 0 and len(laps_leader) - 1 == len(d_laps):
                        l_lapped = 1
                    if d not in lapped_count:
                        lapped_count[d] = l_lapped
                    else:
                        lapped_count[d] = lapped_count[d] + l_lapped

    lapped_count = dict(sorted(lapped_count.items(), key=lambda item: item[1], reverse=True))
    print('LAPS DONE WHILE LAPPED')
    print(lapped_count)


def laps_in_last_place(sessions):
    times_in_last = {}
    for s in sessions:
        if s.name == 'Race' or s.name == 'Sprint':
            laps = (s.laps[s.laps['Position'] == 1]['LapStartDate'].drop_duplicates()
                    .to_frame().sort_values(by='LapStartDate', ascending=True).reset_index(drop=True))
            lap_number = 1
            for i in range(2, len(laps) + 1):

                finish = (s.results[s.results['Status'].astype(str).str.contains(r'Finished|\+', regex=True)]
                          [['Abbreviation', 'Position']].sort_values(by='Position', ascending=True))
                last_driver = finish['Abbreviation'].iloc[len(finish) - 1]

                current_lap = (s.laps.pick_lap(i)[['LapStartDate', 'Driver']]
                               .sort_values(by='LapStartDate', ascending=True))
                if len(current_lap) < len(finish):
                    driver = last_driver
                else:
                    driver = current_lap.iloc[len(current_lap) - 1]['Driver']
                lap_number += 1
                if driver not in times_in_last:
                    times_in_last[driver] = 1
                else:
                    times_in_last[driver] = times_in_last[driver] + 1


    times_in_last = dict(sorted(times_in_last.items(), key=lambda item: item[1], reverse=True))
    print('LAPS IN LAST PLACE')
    print(times_in_last)


def more_times_in_p13(sessions):
    times_p13 = {}
    for s in sessions:
        if 'Practice' in s.name:
            print(s)
            practice_times = {}
            drivers = s.laps['Driver'].unique()
            for d in drivers:
                d_lap = s.laps.pick_driver(d).pick_fastest()['LapTime']
                if not pd.isna(d_lap):
                    practice_times[d] = s.laps.pick_driver(d).pick_fastest()['LapTime']
            practice_times = dict(sorted(practice_times.items(), key=lambda item: item[1], reverse=False))
            if len(list(practice_times.keys())) >= 13:
                driver = list(practice_times.keys())[12]
            else:
                driver = None
        else:
            driver = s.results[s.results['Position'] == 13]['Abbreviation'].iloc[0]
        if driver not in times_p13 and driver is not None:
            times_p13[driver] = 1
        elif driver in times_p13 and driver is not None:
            times_p13[driver] = times_p13[driver] + 1

    times_p13 = dict(sorted(times_p13.items(), key=lambda item: item[1], reverse=True))
    print('MORE TIMES IN P13')
    print(times_p13)


def weather(sessions):
    temp = {}
    for s in sessions:
        min_air = min(s.weather_data['AirTemp'][s.weather_data['AirTemp'] > 0])
        min_track = min(s.weather_data['TrackTemp'][s.weather_data['TrackTemp'] > 0])
        min_humidity = min(s.weather_data['Humidity'][s.weather_data['Humidity'] > 0])
        pressure = min(s.weather_data['Pressure'][s.weather_data['Pressure'] > 0])
        wind_speed = min(s.weather_data['WindSpeed'][s.weather_data['WindSpeed'] > 0])
        temp[s] = {'AIR': min_air, 'TRACK': min_track, 'HUM': min_humidity, 'PRE': pressure, 'WIN': wind_speed}

    temp = dict(sorted(temp.items(), key=lambda item: item[1]['AIR'], reverse=False))
    print('AIR TEMP')
    print(temp)
    temp = dict(sorted(temp.items(), key=lambda item: item[1]['TRACK'], reverse=False))
    print('TRACK TEMP')
    print(temp)
    temp = dict(sorted(temp.items(), key=lambda item: item[1]['HUM'], reverse=False))
    print('min_humidity')
    print(temp)
    temp = dict(sorted(temp.items(), key=lambda item: item[1]['PRE'], reverse=False))
    print('pressure')
    print(temp)
    temp = dict(sorted(temp.items(), key=lambda item: item[1]['WIN'], reverse=False))
    print('wind_speed')
    print(temp)


def awards_2023():
    # sessions = get_all_sessions()
    # with open("awards/sessions.pkl", "wb") as file:
    #     pickle.dump(sessions, file)
    # with open("awards/sessions.pkl", "rb") as file:
    #     sessions = pickle.load(file)
    # # gear_changes_max_RPM(sessions)
    # laps(sessions)
    pit_stops()
    # # off_tracks(sessions)
    # # # slowest_lap(sessions)
    # race_pace_deviation(sessions)
    # # longest_stint(sessions)
    # # times_lapped(sessions)
    # # laps_in_last_place(sessions)
    # # more_times_in_p13(sessions)
    # weather(sessions)
