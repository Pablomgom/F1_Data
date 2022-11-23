import fastf1
from fastf1 import plotting
color_dict = {
    "Ferrari":"#FF0000",
    "Red Bull Racing":"#4B5DEF",
    "Mercedes":"#C5C9C7",
    "Mclaren":"#FFA500",
    "Alpine":"#FF81C0",
    "Alfa Romeo":"#A52A2A",
    "AlphaTauri":"#4d5c6f",
    "Aston Martin":"#006400",
    "Williams":"#OOFFFF",
    "Haas F1 Team":"#FFFFFF"
}

track_status_dict = {
    "1":"Green flag",
    "2":"Yellow flag",
    "4":"SC",
    "5":"Red Flag",
    "6":"VSC deployed",
    "7":"VSC Ending"
}

def init():
    plotting.setup_mpl()
    fastf1.Cache.enable_cache('Cache')