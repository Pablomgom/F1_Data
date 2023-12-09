from src.exceptions.custom_exceptions import QualyException


def year_2023(round):
    if round in [8, 12, 13]:
        raise QualyException