def Red_Bull_Racing_2023(round, max_laps):
    min_laps = 0
    if round == 2:
        min_laps = 20
    if round == 12:
        min_laps = 8
    if round == 14:
        max_laps = 48

    return min_laps, max_laps

def Ferrari_2023(round, max_laps):
    min_laps = 0
    if round == 2:
        min_laps = 20
    if round == 15:
        max_laps = 52
    return min_laps, max_laps

def Mercedes_2023(round, max_laps):

    return 0, max_laps