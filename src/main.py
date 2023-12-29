import sys
import fastf1
import traceback

import pandas as pd
from fastf1.plotting import setup_mpl
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

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
# 1	1	Brazil Ayrton Senna	McLaren-Honda	1:14.210	1:14.041	—
# 2	2	Austria Gerhard Berger	McLaren-Honda	1:14.385	1:15.563	+0.344
# 3	5	United Kingdom Nigel Mansell	Williams-Renault	1:14.822	1:14.897	+0.781
# 4	6	Italy Riccardo Patrese	Williams-Renault	1:15.633	1:15.057	+1.016
# 5	20	Brazil Nelson Piquet	Benetton-Ford	1:16.552	1:15.291	+1.250
# 6	19	Germany Michael Schumacher	Benetton-Ford	1:15.840	1:15.508	+1.467
# 7	28	France Jean Alesi	Ferrari	1:17.014	1:15.545	+1.504
# 8	27	Italy Gianni Morbidelli	Ferrari	1:16.203	1:17.679	+2.162
# 9	4	Italy Stefano Modena	Tyrrell-Honda	1:16.253	45:56.547	+2.212
# 10	23	Italy Pierluigi Martini	Minardi-Ferrari	1:17.614	1:16.359	+2.318
# 11	22	Finland JJ Lehto	Dallara-Judd	1:17.665	1:16.871	+2.830
# 12	33	Italy Andrea de Cesaris	Jordan-Ford	1:17.073	1:17.050	+3.009
# 13	21	Italy Emanuele Pirro	Dallara-Judd	1:17.342	1:18.233	+3.301
# 14	15	Brazil Maurício Gugelmin	Leyton House-Ilmor	1:17.344	1:17.431	+3.303
# 15	9	Italy Michele Alboreto	Footwork-Ford	1:18.214	1:17.355	+3.314
# 16	32	Italy Alessandro Zanardi	Jordan-Ford	1:17.362	1:17.723	+3.321
# 17	8	United Kingdom Mark Blundell	Brabham-Yamaha	1:17.867	1:17.365	+3.324
# 18	24	Brazil Roberto Moreno	Minardi-Ferrari	1:19.752	1:17.639	+3.598
# 19	34	Italy Nicola Larini	Lambo-Lamborghini	1:19.076	1:17.936	+3.895
# 20	25	Belgium Thierry Boutsen	Ligier-Lamborghini	1:18.992	1:17.969	+3.958
# 21	12	United Kingdom Johnny Herbert	Lotus-Judd	1:19.177	1:18.091	+4.050
# 22	26	France Érik Comas	Ligier-Lamborghini	1:19.678	1:18.112	+4.071
# 23	10	Italy Alex Caffi	Footwork-Ford	1:18.783	1:18.157	+4.116
# 24	3	Japan Satoru Nakajima	Tyrrell-Honda	1:18.216	1:18.307	+4.175
# 25	11	Finland Mika Häkkinen	Lotus-Judd	1:19.199	1:18.271	+4.230
# 26	16	Austria Karl Wendlinger	Leyton House-Ilmor	1:18.282	2:12.369	+4.241
# 27	30	Japan Aguri Suzuki	Lola-Ford	26:19.244	1:18.393	+4.352
# 28	7	United Kingdom Martin Brundle	Brabham-Yamaha	1:18.887	1:18.855	+4.814
# 29	35	Belgium Eric van de Poele	Lambo-Lamborghini	1:20.123	1:19.000	+4.959
# 30	29	Belgium Bertrand Gachot	Lola-Ford	1:20.163	1:19.274	+5.233
#
#
#     """
#
#     My_Ergast().insert_qualy_data(1991, 16, data)

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
