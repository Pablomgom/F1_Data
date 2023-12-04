from datetime import timedelta
import pandas as pd
import bar_chart_race as bcr
from fastf1.ergast import Ergast
from src.variables.variables import max_races



def bar_season(year):

    """
         Creates a .mp4 with the points changes in a year

         Parameters:
         year (int): Year to plot
    """

    ergast = Ergast()
    races = ergast.get_race_results(season=year, limit=1000)
    sprints = ergast.get_sprint_results(season=year, limit=1000)
    schedule = ergast.get_race_schedule(season=year, limit=1000)

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

    races_df.sort(key=sort_key)

    all_family_names = set()
    for race in races_df:
        all_family_names.update((race['givenName'] + ' ' + race['familyName']).unique())

    family_points_dict = {name: [] for name in all_family_names}
    for race in races_df:
        current_race_points = {name: 0.0 for name in all_family_names}

        for i in range(len(race)):
            family_name = race.loc[i, 'familyName']
            given_name = race.loc[i, 'givenName']
            full_name = given_name + ' ' + family_name
            points = race.loc[i, 'points']
            current_race_points[full_name] += points
        for name in all_family_names:
            family_points_dict[name].append(current_race_points[name])

    for key in family_points_dict:
        family_points_dict[key].insert(0, 0.0)

    if schedule['season'].min() in max_races.keys():

        races_comput = max_races[schedule['season'].min()]

        for key in family_points_dict:

            sorted_values = sorted(family_points_dict[key], reverse=True)
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
        race_name = races_df[i]['raceName'].min().replace('Sprint','').replace('Grand Prix', 'GP')
        is_sprint = 'Sprint' if races_df[i]['raceType'].min() == 1 else 'Race'

        event = f'Round {round}: {race_name} - {is_sprint}'
        index.append(event)

        if is_sprint == 'Race':
            round += 1

    index.insert(len(races_df), 'Final results')
    for i in range(0,3):
        for key in family_points_dict:
            family_points_dict[key].append(0.0)

        index.insert(len(races_df), 'Final results')

    df = pd.DataFrame(family_points_dict, index=index)
    df.iloc[28, 19] = 6.9
    filename = f'../MP4/F1 Championship - {races.description.season[0]} Season.mp4'
    title = f'F1 Championship - {races.description.season[0]} Season'

    figsize = (1920 / 200, 1080 / 200)
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
            'x': .98,
            'y': .10,
            'ha': 'right',
            'va': 'center',
            'size': 18
        },
        cmap='dark12',
        title=title,
        title_size='xx-large',
        bar_label_size=15,
        tick_label_size=15,
        shared_fontdict={
            'color': '.1',
            'family': 'Fira Sans',
        },
        scale='linear',
        writer=None,
        fig=None,
        bar_kwargs={
            'alpha': .9,
        },
        filter_column_colors=True,
    )