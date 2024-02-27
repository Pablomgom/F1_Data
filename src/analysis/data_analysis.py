import math

import fastf1
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.utils.utils import get_quartiles, find_nearest_non_repeating
from src.variables.team_colors import team_colors_2023, team_colors


def cluster_circuits(year, rounds, prev_year=None, circuit=None, clusters=None, save_data=False):
    """
        Cluster circuits

        Parameters:
        year (int): Year to plot
        rounds (int): Rounds to analyze
        prev_year (int, optional): Take data from a session of the previous year. Default: None
        circuit (str, optional): Circuit to take data from last year. Default: None
        clusters(int, optional): Number of clusters

   """

    session_type = ['FP1', 'FP2', 'FP3', 'Q', 'S', 'SS', 'R']
    circuits = []
    data = []
    if prev_year is None:
        add_round = 0
    else:
        add_round = 1

    for i in range(0, rounds + add_round):
        prev_session = None
        team_fastest_laps = {}
        for type in session_type:
            try:
                if i == rounds:
                    session = fastf1.get_session(prev_year, circuit, type)
                    print(f'{i}: {type}')
                else:
                    session = fastf1.get_session(year, i + 1, type)
                session.load()
                prev_session = session
                teams = set(session.laps['Team'].values)
                for t in teams:
                    fastest_lap = session.laps.pick_team(t).pick_fastest()
                    if fastest_lap['LapTime'] is not np.nan:
                        if t not in team_fastest_laps or fastest_lap['LapTime'] < team_fastest_laps[t]['LapTime']:
                            team_fastest_laps[t] = fastest_lap
                    else:
                        print(f'No data for {t}')
            except Exception as e:
                print(f'{type} not in this event')

        session = prev_session
        corners_location = session.get_circuit_info().corners['Distance'].values
        telemetry_data = []
        speed_in_corners = []
        for lap in team_fastest_laps.values():
            try:
                telemetry = lap.telemetry
                telemetry_data.append(telemetry)
                distance = telemetry['Distance'].values
                corners_in_telemetry = find_nearest_non_repeating(corners_location, distance)
                speed_in_corners.extend(telemetry[telemetry['Distance'].isin(corners_in_telemetry)]['Speed'].values)
            except:
                print('Telemetry error')

        corners = len(session.get_circuit_info().corners)
        length = max(telemetry_data[0]['Distance'])
        corners_per_meter = length / corners
        all_speeds = [max(telemetry['Speed']) for telemetry in telemetry_data]
        all_full_gas = [len(telemetry[telemetry['Throttle'].isin([100, 99])]) / len(telemetry) for telemetry in
                        telemetry_data]
        all_lifting = [len(telemetry[(telemetry['Throttle'] >= 1) & (telemetry['Throttle'] <= 99)]) / len(telemetry) for
                       telemetry in telemetry_data]

        q1_corner_speed = get_quartiles(speed_in_corners)[0]
        median_corner_speed = get_quartiles(speed_in_corners)[1]
        q3_corner_speed = get_quartiles(speed_in_corners)[2]
        avg_corner_speed = get_quartiles(speed_in_corners)[3]

        median_max_speed = np.median(all_speeds)
        median_full_gas = np.median(all_full_gas)
        median_lifting = np.median(all_lifting)

        data.append(
            [corners_per_meter, median_max_speed, median_full_gas, median_lifting,
             q1_corner_speed, median_corner_speed, q3_corner_speed, avg_corner_speed, np.var(speed_in_corners)])

        circuits.append(session.event.Location)


    if save_data:
        data_to_save = pd.DataFrame(data)
        data_to_save.to_csv('../resources/csv/Circuit_data_intermediate.csv', index=False)

    data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    if clusters is None:
        inertia = []
        cluster_range = range(1, 11)
        for clusters in cluster_range:
            kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=42)
            kmeans.fit(principal_components)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, inertia, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()
        exit(0)

    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=33)
    y_kmeans = kmeans.fit_predict(data)

    cluster_means = {}
    for cluster_id in np.unique(y_kmeans):
        cluster_means[cluster_id] = data[y_kmeans == cluster_id].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    texts = []
    for i, name in enumerate(circuits):
        ax.scatter(principal_components[i, 0], principal_components[i, 1], color='#00BFFF', s=100)
        texts.append(ax.text(principal_components[i, 0], principal_components[i, 1], name,
                             font='Fira Sans', fontsize=11.5))

    adjust_text(texts, autoalign='xy', ha='right', va='bottom', only_move={'points': 'y', 'text': 'xy'})
    ax.axis('off')
    ax.grid(False)
    plt.title('SIMILARITY BETWEEN CIRCUITS', font='Fira Sans', fontsize=28)
    plt.tight_layout()
    plt.savefig('../PNGs/Track clusterss.png', dpi=450)
    plt.show()


def predict_race_pace(year=2024, track='Bahrain', session='R'):

    track_features = ['Corner Density', 'Top Speed', 'Full Gas', 'Lifting', 'Q1 Corner Speed',
                      'Median Corner Speed', 'Q3 Corner Speed', 'Average Corner Speed', 'Corner Speed Variance']

    teams = ['Red Bull Racing', 'Aston Martin', 'Alpine', 'Mercedes', 'RB',
             'Kick Sauber', 'Williams', 'Haas F1 Team', 'McLaren', 'Ferrari']

    if session == 'R':
        race_pace_delta = pd.read_csv('../resources/csv/Race_pace_delta.csv')
    else:
        race_pace_delta = pd.read_csv('../resources/csv/Qualy_pace_delta.csv')
    circuit_data = pd.read_csv('../resources/csv/Circuit_data.csv')

    race_pace_delta['Team'] = race_pace_delta['Team'].replace({'AlphaTauri': 'RB',
                                                               'Alfa Romeo': 'Kick Sauber'})
    race_pace_delta.sort_values(by=['Year', 'Session'], inplace=True)
    race_pace_delta['Recent_Avg_Delta'] = race_pace_delta.groupby('Team')['Delta'].transform(
        lambda x: x.ewm(span=3, adjust=False).mean())



    race_pace_delta_pivot = race_pace_delta.pivot_table(index=['Year', 'Track', 'Session'], columns='Team',
                                                        values=['Delta', 'Recent_Avg_Delta']).reset_index()
    race_pace_delta_pivot.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in race_pace_delta_pivot.columns]

    predict_data = pd.merge(race_pace_delta_pivot, circuit_data, on=['Year', 'Track'], how='inner')
    predict_data = predict_data.drop(['Session_y'], axis=1).rename({'Session_x': 'Session'}, axis=1)
    predict_data = predict_data.sort_values(by=['Year', 'Session'], ascending=[True, True]).reset_index(drop=True)
    predict_data['ID'] = range(1, len(predict_data) + 1)
    for team in teams:
        for metric in ['Delta', 'Recent_Avg_Delta']:
            column_name = f'{metric}_{team}'
            if column_name in predict_data.columns:
                median_per_year = predict_data.groupby('Year')[column_name].transform('median')
                predict_data[column_name] = predict_data[column_name].fillna(median_per_year)
    feature_columns = ['Year', 'ID'] + track_features + [f'{metric}_{team}' for team in teams for metric in ['Recent_Avg_Delta']]
    X = predict_data[feature_columns]
    y = predict_data[[f'Delta_{team}' for team in teams]]
    scaler = StandardScaler()
    model = RandomForestRegressor(n_estimators=1000, random_state=42, max_depth=10, max_features='sqrt',
                                  min_samples_leaf=2, min_samples_split=2)
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    prev_year_data = circuit_data[(circuit_data['Year'] == year - 1) & (circuit_data['Track'] == track)]
    if len(prev_year_data) == 0:
        raise Exception('No data for that year/track combination')

    next_race_data = {
        'Year': year,
        'ID': predict_data['ID'].max() + 1
    }
    for feature in track_features:
        next_race_data[feature] = prev_year_data[feature].iloc[0]

    for t in teams:
        next_race_data[f'Recent_Avg_Delta_{t}'] = predict_data[f'Recent_Avg_Delta_{t}'].values[-1]

    next_race_df = pd.DataFrame([next_race_data])
    next_race_data_scaled = scaler.transform(next_race_df)
    next_race_prediction = model.predict(next_race_data_scaled)
    min_value = min(next_race_prediction[0])
    adjusted_predictions = [value - min_value for value in next_race_prediction[0]]
    next_race_prediction = pd.DataFrame([adjusted_predictions], columns=teams)
    next_race_prediction = next_race_prediction.transpose().sort_values(by=0, ascending=False)
    ref_session = fastf1.get_session(year - 1, track, session)
    ref_session.load()
    fastest_team = next_race_prediction.index[len(next_race_prediction) - 1]
    delta_time = ref_session.laps.pick_team(fastest_team).pick_quicklaps().pick_wo_box()['LapTime'].median()
    if session == 'Q':
        delta_time = ref_session.laps.pick_team(fastest_team).pick_fastest()['LapTime']

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [team_colors.get(year)[t] for t in next_race_prediction.index]
    bars = plt.barh(next_race_prediction.index, next_race_prediction[0].values, color=colors)
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.02
        diff_in_seconds = round(((width/100) * delta_time).total_seconds(), 3)
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}% (+{diff_in_seconds:.3f}s)',
                va='center', ha='left', font='Fira Sans', fontsize=16)
    if session == 'R':
        plt.title(f'{track.upper()} {year} RACE PACE SIMULATION', font='Fira Sans', fontsize=28)
    else:
        plt.title(f'{track.upper()} {year} QUALY SIMULATION', font='Fira Sans', fontsize=30)
    plt.yticks(font='Fira Sans', fontsize=16)
    plt.xticks(font='Fira Sans', fontsize=16)
    plt.xlabel('Percentage difference', font='Fira Sans', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{year} {track} {session} PREDICTION.png', dpi=450)
    plt.show()


