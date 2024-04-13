import sys
from time import sleep

import fastf1
import traceback

import pandas as pd
from fastf1.plotting import setup_mpl
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

import src.utils.utils
from src.analysis.drivers import race_qualy_h2h
from src.awards.awards import awards_2023
from src.db.db import Database
from src.ergast_api.my_ergast import My_Ergast
from src.menu.menu import get_funcs

from src.utils.utils import parse_args, is_session_first_arg

setup_mpl(misc_mpl_mods=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
fastf1.ergast.interface.BASE_URL = 'http://ergast.com/api/f1'
plt.rcParams["font.family"] = "Fira Sans"
fastf1.Cache.enable_cache('../cache')

FUNCTION_MAP = get_funcs()
session = None
previous_input = ""

if __name__ == '__main__':


#     data = """
#
# 1//1//Max Verstappen//Red Bull Racing-Honda RBPT//1:28.866//1:28.740//1:28.197//1
# 2//11//Sergio Pérez//Red Bull Racing-Honda RBPT//1:29.303//1:28.752//1:28.263//2
# 3//4//Lando Norris//McLaren-Mercedes//1:29.536//1:28.940//1:28.489//3
# 4//55//Carlos Sainz//Ferrari//1:29.513//1:29.099//1:28.682//4
# 5//14//Fernando Alonso//Aston Martin Aramco-Mercedes//1:29.254//1:29.082//1:28.686//5
# 6//81//Oscar Piastri//McLaren-Mercedes//1:29.425//1:29.148//1:28.760//6
# 7//44//Lewis Hamilton//Mercedes//1:29.661//1:28.887//1:28.766//7
# 8//16//Charles Leclerc//Ferrari//1:29.338//1:29.196//1:28.786//8
# 9//63//George Russell//Mercedes//1:29.799//1:29.140//1:29.008//9
# 10//22//Yuki Tsunoda//RB-Honda RBPT//1:29.775//1:29.417//1:29.413//10
# 11//3//Daniel Ricciardo//RB-Honda RBPT//1:29.727//1:29.472//N/A//11
# 12//27//Nico Hülkenberg//Haas-Ferrari//1:29.821//1:29.494//N/A//12
# 13//77//Valtteri Bottas//Kick Sauber-Ferrari//1:29.602//1:29.593//N/A//13
# 14//23//Alexander Albon//Williams-Mercedes//1:29.963//1:29.714//N/A//14
# 15//31//Esteban Ocon//Alpine-Renault//1:29.811//1:29.816//N/A//15
# 16//18//Lance Stroll//Aston Martin Aramco-Mercedes//1:30.024//N/A//N/A//16
# 17//10//Pierre Gasly//Alpine-Renault//1:30.119//N/A//N/A//17
# 18//20//Kevin Magnussen//Haas-Ferrari//1:30.131//N/A//N/A//18
# 19//2//Logan Sargeant//Williams-Mercedes//1:30.139//N/A//N/A//19
# 20//24//Guanyu Zhou//Kick Sauber-Ferrari//1:30.143//N/A//N/A//20
# """
#
#     My_Ergast().insert_qualy_data(2024, 4, data,
#                                   offset=0, character_sep='-', have_country=False, number=1)




    while True:
        func = input(f"Enter the function name (or 'exit' to quit) [{previous_input}]: ")
        func_name = func.split('(')[0]
        if func_name.lower() == 'exit':
            print("Exiting...")
            sys.exit()
        try:
            args_input = func.split('(')[1].replace(')', '')
            args, kwargs = parse_args(args_input, FUNCTION_MAP, session)
            func = FUNCTION_MAP[func_name]
            if is_session_first_arg(func):
                args.insert(0, session)
            result = FUNCTION_MAP[func_name](*args, **kwargs)
            if func_name.lower() == 'load_session':
                session = result
            if result is not None:
                print(f"Result: {result}")
        except Exception as e:
            traceback.print_exc()
            sleep(0.2)
