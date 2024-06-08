import math
import warnings

import fastf1
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt
from scipy import stats
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
import matplotlib.patheffects as path_effects
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    median_absolute_error, euclidean_distances

from src.plots.plots import round_bars, annotate_bars
from src.utils.utils import get_quartiles, find_nearest_non_repeating
from src.variables.team_colors import team_colors_2023, team_colors, team_colors_2024


def cluster_circuits(year, rounds, prev_year=None, circuit=None, clusters=3, save_data=False):
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
        add_round = 2

    for i in range(0, rounds + add_round):
        prev_session = None
        team_fastest_laps = {}
        for type in session_type:
            try:
                if i == rounds:
                    session = fastf1.get_session(prev_year, circuit, type)
                    print(f'{i}: {type}')
                else:
                    session = fastf1.get_session(year, 'Monaco', type)
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
    plt.savefig(f'../PNGs/Track clusters {year}.png', dpi=450)
    plt.show()


def predict_race_pace(year=2024, track='Bahrain', session='R'):
    track_features = ['Corner Density', 'Top Speed', 'Full Gas', 'Lifting', 'Q1 Corner Speed',
                      'Median Corner Speed', 'Q3 Corner Speed', 'Average Corner Speed', 'Corner Speed Variance']

    teams = ['Red Bull Racing', 'Aston Martin', 'Alpine', 'Mercedes', 'RB',
             'Kick Sauber', 'Williams', 'Haas F1 Team', 'McLaren', 'Ferrari']

    ewm_cols = ['Team', 'Year']
    ordinal_mapping = {'Low': 1, 'Medium': 2, 'High': 3}

    if session == 'R':
        race_pace_delta = pd.read_csv('../resources/csv/Race_pace_delta.csv')
    else:
        race_pace_delta = pd.read_csv('../resources/csv/Qualy_pace_delta.csv')
        race_pace_delta = race_pace_delta[~pd.isna(race_pace_delta['Delta'])]

    circuit_data = pd.read_csv('../resources/csv/Circuit_data.csv')
    for t in track_features:
        if t == 'Corner Density':
            circuit_data[t] = pd.qcut(circuit_data[t], q=3, labels=['High', 'Medium', 'Low'])
        else:
            circuit_data[t] = pd.qcut(circuit_data[t], q=3, labels=['Low', 'Medium', 'High'])
        circuit_data[t] = circuit_data[t].map(ordinal_mapping)

    race_pace_delta['Team'] = race_pace_delta['Team'].replace({'AlphaTauri': 'RB',
                                                               'Alfa Romeo': 'Kick Sauber'})
    race_pace_delta.sort_values(by=['Year', 'Session'], inplace=True)
    race_pace_delta['Recent_Avg_Delta'] = race_pace_delta.groupby(ewm_cols)['Delta'].transform(
        lambda x: x.ewm(span=3, adjust=False).mean())

    race_pace_delta_pivot = race_pace_delta.pivot_table(index=['Year', 'Track', 'Session'],
                                                        columns='Team',
                                                        values=['Delta', 'Recent_Avg_Delta']
                                                        ).reset_index()
    race_pace_delta_pivot.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in race_pace_delta_pivot.columns]
    predict_data = pd.merge(race_pace_delta_pivot, circuit_data, on=['Year', 'Track'], how='inner')
    predict_data = predict_data.drop(['Session_y'], axis=1).rename({'Session_x': 'Session'}, axis=1)
    predict_data = predict_data.sort_values(by=['Year', 'Session'], ascending=[True, True]).reset_index(drop=True)
    predict_data['ID'] = range(1, len(predict_data) + 1)
    for team in teams:
        for feature in track_features:
            delta_column = f'Delta_{team}'
            if delta_column in predict_data.columns:
                interaction_column_name = f'{feature}*Delta_{team}'
                predict_data[interaction_column_name] = predict_data[feature].astype(float) * predict_data[
                    delta_column].astype(float)
            else:
                print(f"Column {delta_column} not found in DataFrame.")

    cols_to_predict = [f'Delta_{team}' for team in teams]
    X = predict_data[[c for c in predict_data.columns if c not in cols_to_predict and c not in ['Track', 'Session']]]
    y = predict_data[cols_to_predict]
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    lasso = Lasso(alpha=0.007543120063354615)
    param_grid = {'alpha': np.logspace(-4, 0, 50)}
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    grid_search.fit(X_train, y_train)
    print("Best alpha: ", grid_search.best_params_['alpha'])
    best_lasso = grid_search.best_estimator_
    test_score = best_lasso.score(X_test, y_test)
    print("Test set score: ", test_score)
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    used_model = Lasso(alpha=grid_search.best_params_['alpha'])
    used_model.fit(X_train, y_train)

    y_pred = used_model.predict(X_test)
    error_mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    error_rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    cv_rmse = rmse / np.mean(y_test)
    mbe = np.mean(y_pred - y_test)
    n = len(y_test)  # number of samples
    p = X_test.shape[1]  # number of independent variables
    r2 = used_model.score(X_test, y_test)  # assuming 'used_model' is your model variable
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    r2 = used_model.score(X_test, y_test)

    print(f"""ERRORS:
                MAE: {error_mae}
                MEDAE: {medae}
                CV(RMSE): {cv_rmse}
                RMSE: {error_rmse}  
                MBE: {mbe}
                R2: {r2} 
                ADJUSTED R2: {adjusted_r2}
        """)

    prev_year_data = circuit_data[(circuit_data['Year'] == year - 1) & (circuit_data['Track'] == track)]
    if len(prev_year_data) == 0:
        raise Exception('No data for that year/track combination')

    def find_similar_tracks(target_features, historical_data, track_features, top_n=3, current_year=year):
        # relevant_years_data = historical_data[historical_data['Year'].isin([current_year])]
        # distances = euclidean_distances(relevant_years_data[track_features], target_features)
        # recency_factor = 1 / (current_year - relevant_years_data['Year'] + 1)
        # adjusted_distances = distances.flatten() * recency_factor
        # relevant_years_data['adjusted_similarity_score'] = adjusted_distances
        # similar_tracks = relevant_years_data.sort_values(by='adjusted_similarity_score')
        # similar_tracks['recency_factor'] = 1
        # final_similar_tracks = similar_tracks.head(top_n)

        last_three_sessions = historical_data.sort_values(by=['Year', 'Session'], ascending=[False, False]).head(top_n)
        max_session = max(last_three_sessions['Session'])
        # combined_results = pd.concat([final_similar_tracks, last_three_sessions])
        combined_results = last_three_sessions
        distances = euclidean_distances(combined_results[track_features], target_features)
        recency_factor = 1 / (current_year - combined_results['Year'] + 1)
        adjusted_distances = distances.flatten() * recency_factor
        combined_results['adjusted_similarity_score'] = adjusted_distances
        return combined_results

    def calculate_additional_features(similar_tracks, teams, track_features, predict_data, current_year=year):
        calculated_features = {}
        circuit_recent_data = pd.merge(predict_data, similar_tracks, 'inner', ['Year', 'Track'])
        circuit_recent_data['year_weight'] = circuit_recent_data['Year'].apply(
            lambda x: 1)

        max_score = circuit_recent_data['adjusted_similarity_score'].mean()
        circuit_recent_data['weight'] = 1 / (circuit_recent_data['adjusted_similarity_score'] + max_score)
        circuit_recent_data['final_weight'] = circuit_recent_data['weight'] * circuit_recent_data['year_weight']

        for team in teams:
            for feature in track_features + ['Recent_Avg_Delta']:
                if feature == 'Recent_Avg_Delta':
                    feature_name = f'Recent_Avg_Delta_{team}'
                else:
                    feature_name = f"{feature}*Delta_{team}"

                weighted_sum = (circuit_recent_data[feature_name] * circuit_recent_data['final_weight']).sum()
                total_weight = circuit_recent_data['final_weight'].sum()
                calculated_value = weighted_sum / total_weight
                calculated_features[feature_name] = calculated_value

        return calculated_features

    def scale_features(prediction_data, scaler):
        return scaler.transform(prediction_data)

    def make_prediction(model, prediction_data):
        y_pred = model.predict(prediction_data)
        return y_pred

    def prepare_prediction_data(similar_tracks, additional_features, model_features, track_features):
        basic_aggregated_data = similar_tracks[track_features].astype(float).mean()
        prediction_data = {**basic_aggregated_data.to_dict(), **additional_features}
        prediction_data_ordered = {feature: prediction_data.get(feature, 0) for feature in model_features}
        return pd.DataFrame([prediction_data_ordered])

    upcoming_race_features = prev_year_data[track_features]
    similar_tracks = find_similar_tracks(upcoming_race_features, circuit_data, track_features)
    additional_features = calculate_additional_features(similar_tracks, teams, track_features, predict_data)
    prediction_data = prepare_prediction_data(similar_tracks, additional_features,
                                              [c for c in predict_data.columns if c not in cols_to_predict
                                               and c not in ['Track', 'Session']], track_features)
    prediction_data['Year'] = year
    prediction_data['ID'] = predict_data['ID'].max() + 1
    prediction_data_scaled = scale_features(prediction_data, scaler)
    next_race_prediction = make_prediction(used_model, prediction_data_scaled)

    min_value = min(next_race_prediction[0])
    adjusted_predictions = [value - min_value for value in next_race_prediction[0]]
    next_race_prediction = pd.DataFrame([adjusted_predictions], columns=teams)
    next_race_prediction = next_race_prediction.transpose().sort_values(by=0, ascending=False)
    print(next_race_prediction)
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
        diff_in_seconds = round(((width / 100) * delta_time).total_seconds(), 3)
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


def performance_by_turns(session, track):
    year = session.event.year
    drivers = set(session.laps['Driver'])
    turns = pd.read_csv('../resources/csv/Turn_analysis.csv')
    turns = turns[turns['Track'] == track]
    turn_names = turns['Name'].values
    turns_time = pd.DataFrame(columns=['Turn', 'Team', 'Driver', 'Time'])
    for d in drivers:
        for index, lap in session.laps.pick_driver(d).pick_not_deleted().iterlaps():
            try:
                team_lap = lap
                team = team_lap['Team']
                team_tel = pd.DataFrame(team_lap.get_telemetry()[['Distance', 'Speed', 'Time', 'Throttle']])
                new_distance = np.linspace(team_tel['Distance'].min(), team_tel['Distance'].max(), 50000)
                interpolated_data = {'Distance': new_distance}
                team_tel['Time'] = team_tel['Time'].transform(lambda x: x.total_seconds())
                for column in team_tel.columns:
                    if column != 'Distance':
                        cs = CubicSpline(team_tel['Distance'], team_tel[column])
                        interpolated_data[column] = cs(new_distance)
                team_tel = pd.DataFrame(interpolated_data)

                for index, turn in turns.iterrows():
                    start_time = np.interp(turn['Start'], team_tel['Distance'], team_tel['Time'])
                    end_time = np.interp(turn['End'], team_tel['Distance'], team_tel['Time'])
                    total_time = end_time - start_time
                    turn_data = pd.DataFrame([[turn['Name'], team, d, total_time]], columns=turns_time.columns)
                    turns_time = pd.concat([turns_time, turn_data], ignore_index=True)
            except ValueError:
                print(f'No valid lap for {d}')
    fastest_entries_index = turns_time.groupby(['Turn', 'Team'])['Time'].idxmin()
    turns_time = turns_time.loc[fastest_entries_index]
    turns_time['Time Difference'] = turns_time['Time'] - turns_time.groupby('Turn')['Time'].transform(min)
    for name in turn_names:
        df_to_plot = turns_time[turns_time['Turn'] == name].sort_values(by='Time Difference').reset_index(drop=True)
        print(name)
        for index, row in df_to_plot.iterrows():
            print(f'{index + 1}: {row["Team"]} (+{row["Time Difference"]:.3f}s)')
        fix, ax = plt.subplots(figsize=(9, 8))
        bars = plt.bar(df_to_plot['Team'], df_to_plot['Time Difference'])
        colors = [team_colors[year].get(i) for i in df_to_plot['Team']]
        round_bars(bars, ax, colors, color_1=None, color_2=None, y_offset_rounded=0.03, corner_radius=0.08, linewidth=4)
        annotate_bars(bars, ax, 0.003, 14, text_annotate='+{height}s', ceil_values=False, round=3,
                      y_negative_offset=0.04, annotate_zero=False, negative_offset=0)

        plt.title(f'PERFORMANCE IN {track.upper()} - {name.upper()}', font='Fira Sans', fontsize=24)
        plt.xlabel('Team', font='Fira Sans', fontsize=20)
        plt.ylabel('Diff to fastest (s)', font='Fira Sans', fontsize=20)
        plt.xticks(rotation=90, font='Fira Sans', fontsize=18)
        plt.yticks(font='Fira Sans', fontsize=16)
        plt.ylim(bottom=0, top=max(df_to_plot['Time Difference']) + 0.1)
        color_index = 0
        for label in ax.get_xticklabels():
            label.set_color('white')
            label.set_fontsize(16)
            for_color = colors[color_index]
            if for_color == '#ffffff':
                for_color = '#FF7C7C'
            label.set_path_effects([path_effects.withStroke(linewidth=2, foreground=for_color)])
            color_index += 1
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'../PNGs/PERFORMANCE IN {track.upper()} - {name}.png', dpi=450)
        plt.show()
