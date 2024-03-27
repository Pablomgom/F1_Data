import math

import fastf1
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt
from scipy import stats
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
import matplotlib.patheffects as path_effects
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    median_absolute_error

from src.plots.plots import round_bars, annotate_bars
from src.utils.utils import get_quartiles, find_nearest_non_repeating
from src.variables.team_colors import team_colors_2023, team_colors, team_colors_2024


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


    coefficients = best_lasso.coef_
    for target_index in range(coefficients.shape[0]):
        target_coefficients = coefficients[target_index, :]
        coefficients_series = pd.Series(target_coefficients,
                                        index=[c for c in predict_data.columns if c not in cols_to_predict and c not in ['Track', 'Session']])
        plt.figure(figsize=(14, 12))
        coefficients_series.plot(kind='bar')
        plt.title(f'Feature Importance for Target {target_index + 1}')
        plt.xlabel('Feature')
        plt.ylabel('Coefficient Magnitude')
        plt.tight_layout()
        plt.show()

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
        for c in predict_data.columns:
            if ('Recent' in c or '*' in c) and t in c:
                next_race_data[c] = predict_data[c].values[-1]

    next_race_df = pd.DataFrame([next_race_data])[X.columns]
    next_race_data_scaled = scaler.transform(next_race_df)
    next_race_prediction = used_model.predict(next_race_data_scaled)
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



def performance_by_turns(session):

    teams = set(session.laps['Team'])
    turns = session.get_circuit_info().corners[['Number', 'Distance']]
    circuit_data = pd.DataFrame(session.laps.pick_fastest().get_telemetry()[['X', 'Y', 'Throttle', 'Distance']])
    dx = np.diff(circuit_data['X'])
    dy = np.diff(circuit_data['Y'])
    angles = np.arctan2(dy, dx)
    angle_changes = np.abs(np.diff(np.unwrap(angles)))
    angle_change_threshold = 0.04
    straight_segments = angle_changes < angle_change_threshold
    straight_segments_with_start = np.insert(straight_segments, 0, [True, True])
    window_size = 10
    kernel = np.ones(window_size) / window_size
    smoothed_segments = np.convolve(straight_segments_with_start, kernel, mode='same')
    threshold = 0.5
    straight_segments_with_start_s = smoothed_segments > threshold
    change_indices = np.where(np.diff(straight_segments_with_start_s))[0] + 1
    all_indices = np.concatenate(([0], change_indices, [straight_segments_with_start_s.size]))
    segment_lengths = np.diff(all_indices)
    for i, (start, length) in enumerate(zip(all_indices[:-1], segment_lengths), 1):
        if straight_segments_with_start_s[start] == False and length < 10:
            if (i == 1 or straight_segments_with_start_s[all_indices[i - 2]] == True) and (
                    i == len(segment_lengths) or straight_segments_with_start_s[all_indices[i]] == True):
                straight_segments_with_start_s[start:start + length] = True

    for i in range(len(straight_segments_with_start_s)):
        if circuit_data['Throttle'].reset_index(drop=True)[i] != 100:
            straight_segments_with_start_s[i] = False

    plt.figure(figsize=(10, 6))
    for i in range(len(circuit_data) - 1):
        if straight_segments_with_start_s[i]:
            plt.plot(circuit_data['X'].iloc[i:i + 2], circuit_data['Y'].iloc[i:i + 2], 'g-')
        else:
            plt.plot(circuit_data['X'].iloc[i:i + 2], circuit_data['Y'].iloc[i:i + 2], 'r-')

    plt.grid(True)
    plt.axis('equal')
    # plt.show()

    straight_points = np.where(straight_segments_with_start_s)[0]
    diffs = np.diff(straight_points)
    breaks = np.where(diffs > 1)[0] + 1
    segments = np.split(straight_points, breaks)

    filtered_segments = []
    for segment in segments:
        if len(segment) > 25:  # Check if the segment length is more than 10
            trimmed_segment = segment[8:-2]  # Remove first and last 5 elements
            filtered_segments.append(trimmed_segment)

    segment_distances = []

    for segment in filtered_segments:
        if len(segment) > 0:
            first_last_distances = (circuit_data['Distance'].iloc[segment[0]], circuit_data['Distance'].iloc[segment[-1]])
            segment_distances.append(first_last_distances)

    turns_speeed = pd.DataFrame(columns=['Turn', 'Team', 'Driver', 'Speed'])
    turns_time = pd.DataFrame(columns=['Turn', 'Team', 'Driver', 'Time'])
    straight_zone_data = pd.DataFrame(columns=['Team', 'Driver', 'Time'])
    for t in teams:
        team_lap = session.laps.pick_team(t).pick_fastest()
        driver = team_lap['Driver']
        team_tel = pd.DataFrame(team_lap.get_telemetry()[['Distance', 'Speed', 'Time', 'Throttle']])
        new_distance = np.linspace(team_tel['Distance'].min(), team_tel['Distance'].max(), 50000)
        interpolated_data = {'Distance': new_distance}
        team_tel['Time'] = team_tel['Time'].transform(lambda x: x.total_seconds())

        for column in team_tel.columns:
            if column != 'Distance':
                cs = CubicSpline(team_tel['Distance'], team_tel[column])
                interpolated_data[column] = cs(new_distance)
        team_tel = pd.DataFrame(interpolated_data)

        max_distance = max(team_tel['Distance'])
        for index, row in turns.iterrows():
            turn_distance = row[1]
            start_point = max(0, turn_distance - 25)
            end_point = min(max_distance, turn_distance + 25)
            start_time = np.interp(start_point, team_tel['Distance'], team_tel['Time'])
            end_time = np.interp(end_point, team_tel['Distance'], team_tel['Time'])

            lower_bound = turn_distance - 30
            upper_bound = turn_distance + 30
            range_df = team_tel[(team_tel['Distance'] >= lower_bound) & (team_tel['Distance'] <= upper_bound)]
            min_speed_value = stats.trim_mean(range_df['Speed'], 0.03)

            turn_data = pd.DataFrame([[row[0], t, driver, min_speed_value]], columns=turns_speeed.columns)
            turns_speeed = pd.concat([turns_speeed, turn_data], ignore_index=True)
            time_delta = end_time - start_time
            team_turn_data = pd.DataFrame([[row[0], t, driver, time_delta]], columns=turns_time.columns)
            turns_time = pd.concat([turns_time, team_turn_data], ignore_index=True)

        for straight_zone in segment_distances:
            start = straight_zone[0]
            end = straight_zone[1]
            start_time = np.interp(start, team_tel['Distance'], team_tel['Time'])
            end_time = np.interp(end, team_tel['Distance'], team_tel['Time'])
            time_delta = end_time - start_time
            team_straight_data = pd.DataFrame([[t, driver, time_delta]], columns=straight_zone_data.columns)
            straight_zone_data = pd.concat([straight_zone_data, team_straight_data], ignore_index=True)

    speed_dict = {'Low': 140, 'Medium': 210}
    turns_speeed_category = turns_speeed.groupby('Turn')['Speed'].median().reset_index()
    turns_speeed_category = turns_speeed_category.rename(columns={'Speed': 'Median Speed Turn'})

    def categorize_speed(speed):
        if speed < speed_dict['Low']:
            return 'Low'
        elif speed < speed_dict['Medium']:
            return 'Medium'
        else:
            return 'High'

    turns_speeed_category['Category'] = turns_speeed_category['Median Speed Turn'].apply(categorize_speed)
    turns_speeed = pd.merge(turns_speeed, turns_speeed_category, on='Turn', how='inner')
    turns_exceptions = pd.read_csv('../resources/csv/Turns_exceptions.csv')
    location = session.event.Location
    year = session.event.year
    turns_exceptions = turns_exceptions[turns_exceptions['Year'] == year]
    turns_exceptions = turns_exceptions[turns_exceptions['Track'] == location]
    turns_speeed_filtered = pd.merge(turns_speeed, turns_exceptions, on='Turn', how='left', indicator=True)
    turns_speeed_filtered = turns_speeed_filtered[turns_speeed_filtered['_merge'] == 'left_only']
    print(turns_speeed_filtered['Turn'].drop_duplicates())
    turns_speeed_filtered = (turns_speeed_filtered.groupby(['Team', 'Category'])['Speed'].median().reset_index()
                            .sort_values(['Category', 'Speed'], ascending=[True, False]))
    print(turns_speeed_filtered)

    unique_categories = turns_speeed_filtered['Category'].unique()
    dfs_by_category = {}
    for category in unique_categories:
        dfs_by_category[category] = turns_speeed_filtered[turns_speeed_filtered['Category'] == category]

    for category in unique_categories:
        df_to_plot = dfs_by_category[category]
        fix, ax = plt.subplots(figsize=(9, 8))
        bars = plt.bar(df_to_plot['Team'], df_to_plot['Speed'])
        colors = [team_colors_2024.get(i) for i in df_to_plot['Team']]
        round_bars(bars, ax, colors, color_1=None, color_2=None, y_offset_rounded=0.03, corner_radius=0.1, linewidth=4)
        annotate_bars(bars, ax, 0.1, 14, text_annotate='default', ceil_values=False, round=2,
                      y_negative_offset=0.04, annotate_zero=False, negative_offset=0)

        plt.title(f'PERFORMANCE IN {category.upper()} SPEED CORNERS', font='Fira Sans', fontsize=24)
        plt.xlabel('Team', font='Fira Sans', fontsize=20)
        plt.ylabel('Average speed', font='Fira Sans', fontsize=20)
        plt.xticks(rotation=90, font='Fira Sans', fontsize=18)
        plt.yticks(font='Fira Sans', fontsize=16)
        plt.ylim(bottom=min(df_to_plot['Speed']) - 5, top=max(df_to_plot['Speed']) + 5)
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
        plt.savefig(f'../PNGs/PERFORMANCE IN {category.upper()} SPEED CORNERS.png', dpi=450)
        plt.show()
