from datetime import timedelta

import fastf1
import pandas as pd
import bar_chart_race as bcr
from matplotlib import pyplot as plt, animation

from src.variables.variables import max_races



def bar_race(races, sprints, schedule):

    puntos = {}
    races_df = []

    for i in range(len(races.content)):
        df = races.content[i]
        df = pd.DataFrame(df)
        df['raceType'] = 0
        df['raceDate'] = races.description.loc[i, 'raceDate']
        df['raceName'] = races.description.loc[i, 'raceName']
        races_df.append(df)

    for i in range(len(sprints.content)):
        df = sprints.content[i]
        df = pd.DataFrame(df)
        df['raceType'] = 1
        df['raceDate'] = sprints.description.loc[i, 'raceDate'] - timedelta(days=1)
        df['raceName'] = sprints.description.loc[i, 'raceName'] + 'Sprint'
        races_df.append(df)

    def sort_key(df):
        return df['raceDate'].min(), -df['raceType'].max()

    # Sort the list of dataframes
    races_df.sort(key=sort_key)

    all_family_names = set()
    for race in races_df:
        all_family_names.update(race['familyName'].unique())

    # initialize dictionary
    family_points_dict = {name: [] for name in all_family_names}

    # iterate over the races
    for race in races_df:
        # add zero points for all family names for current race
        current_race_points = {name: 0.0 for name in all_family_names}

        for i in range(len(race)):
            family_name = race.loc[i, 'familyName']
            points = race.loc[i, 'points']
            current_race_points[family_name] += points

        # add points from the current race to the total points
        for name in all_family_names:
            family_points_dict[name].append(current_race_points[name])

    for key in family_points_dict:
        family_points_dict[key].insert(0, 0.0)

    if schedule['season'].min() in max_races.keys():

        races_comput = max_races[schedule['season'].min()]

        for key in family_points_dict:
            sorted_values = sorted(family_points_dict[key], reverse=True)

            # Get the 4th highest value
            if len(sorted_values) >= races_comput:
                threshold = sorted_values[races_comput - 1]
            else:
                threshold = min(sorted_values)

            changes = 1
            new_points = []

            for points in family_points_dict[key]:
                if points >= threshold and points > 0.0:
                    if changes <= races_comput:
                        new_points.append(points)
                        changes += 1
                    else:
                        if points > threshold and threshold in new_points:
                            index = len(new_points) - 1 - new_points[::-1].index(threshold)
                            new_points[index] = 0.0
                            new_points.append(points)
                        else:
                            new_points.append(0.0)
                else:
                    new_points.append(0.0)

            family_points_dict[key] = new_points

    index = []
    round = 1
    for i in range(len(races_df)):
        date = races_df[i]['raceDate'].min().strftime("%Y-%m-%d")
        race_name = races_df[i]['raceName'].min().replace('Sprint','')
        is_sprint = 'Sprint' if races_df[i]['raceType'].min() == 1 else 'Race'

        event = f'Round {round} - {is_sprint} - {race_name} - {date}'
        index.append(event)

        if is_sprint == 'Race':
            round += 1

    index.insert(len(races_df), 'Final results')
    index.insert(len(races_df), 'Final results')

    for key in family_points_dict:
        family_points_dict[key].append(0.0)

    # Create a sample DataFrame.
    df = pd.DataFrame(family_points_dict, index=index)

    filename = f'../MP4/F1 Championship - {races.description.season[0]} Season.mp4'
    title = f'F1 Championship - {races.description.season[0]} Season'

    figsize = (1920 / 200, 1080 / 200)
    # Create a bar chart race, save as .mp4.
    bcr.bar_chart_race(
        df=df.cumsum(),
        filename=filename,
        figsize=figsize,
        dpi=200,
        period_length=2000,  # 60fps
        orientation='h',
        sort='desc',
        n_bars=10,
        fixed_order=False,
        fixed_max=True,
        steps_per_period=100,
        interpolate_period=False,
        label_bars=True,
        bar_size=.95,
        period_label={
            'x': .95,
            'y': .10,
            'ha': 'right',
            'va': 'center',
            'size': 11
        },
        cmap='dark12',
        title=title,
        title_size='',
        bar_label_size=11,
        tick_label_size=11,
        shared_fontdict={
            'color': '.1'
        },
        scale='linear',
        writer=None,
        fig=None,
        bar_kwargs={
            'alpha': .7,
        },
        filter_column_colors=True,
    )

