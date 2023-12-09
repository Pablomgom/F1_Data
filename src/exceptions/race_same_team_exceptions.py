from src.exceptions.custom_exceptions import RaceException


def Red_Bull_Racing_2023(round, max_laps=0):
    min_laps = 0
    if round == 2:
        min_laps = 20
    if round == 12:
        min_laps = 8
    if round == 14:
        max_laps = 48

    return min_laps, max_laps


def Ferrari_2023(round, max_laps=0):
    min_laps = 0
    if round == 12:
        raise RaceException
    if round == 13:
        raise RaceException
    if round == 15:
        max_laps = 52
    if round == 21:
        raise RaceException
    return min_laps, max_laps


def Mercedes_2023(round, max_laps=0):
    return 0, max_laps

def Red_Bull_Racing_2021(round, max_laps=0):
    min_laps = 0
    if round == 11:
        raise RaceException

    return min_laps, max_laps

def Mercedes_2021(round, max_laps=0):
    return 0, max_laps


def McLaren_2023(round, max_laps=0):
    if round == 9:
        raise RaceException
    if round == 15:
        raise RaceException
    return 0, max_laps


def Aston_Martin_2023(round, max_laps=0):
    return 0, max_laps


def Alpine_2023(round, max_laps=0):
    return 0, max_laps

def Alpine_2022(round, max_laps=0):
    return 0, max_laps


def Williams_2023(round, max_laps=0):
    return 0, max_laps


def Alfa_Romeo_2023(round, max_laps=0):
    return 0, max_laps


def Haas_F1_Team_2023(round, max_laps=0):
    return 0, max_laps

def AlphaTauri_2023(round, max_laps=0):
    return 0, max_laps