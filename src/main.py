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

# setup_mpl(misc_mpl_mods=False)
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
# 1	1	Brazil Nelson Piquet	Brabham-BMW	1:30.889	1:21.703	 —
# 2	7	France Alain Prost	McLaren-TAG	1:28.276	1:21.774	+0.071
# 3	19	Brazil Ayrton Senna	Toleman-Hart	1:30.077	1:21.936	+0.233
# 4	6	Finland Keke Rosberg	Williams-Honda	1:32.269	1:22.049	+0.346
# 5	11	Italy Elio de Angelis	Lotus-Renault	1:28.428	1:22.291	+0.588
# 6	12	United Kingdom Nigel Mansell	Lotus-Renault	1:32.986	1:22.319	+0.616
# 7	15	France Patrick Tambay	Renault	1:29.409	1:22.583	+0.880
# 8	27	Italy Michele Alboreto	Ferrari	1:31.192	1:22.686	+0.983
# 9	16	United Kingdom Derek Warwick	Renault	1:35.913	1:22.801	+1.098
# 10	20	Sweden Stefan Johansson	Toleman-Hart	1:28.991	1:22.942	+1.239
# 11	8	Austria Niki Lauda	McLaren-TAG	1:28.837	1:23.183	+1.480
# 12	22	Italy Riccardo Patrese	Alfa Romeo	1:37.154	1:24.048	+2.345
# 13	33	France Philippe Streiff	Renault	1:37.280	1:24.089	+2.386
# 14	23	United States Eddie Cheever	Alfa Romeo	1:34.809	1:24.235	+2.532
# 15	5	France Jacques Laffite	Williams-Honda	1:39.696	1:24.437	+2.734
# 16	17	Switzerland Marc Surer	Arrows-BMW	1:34.003	1:24.688	+2.985
# 17	28	France René Arnoux	Ferrari	1:36.634	1:24.848	+3.145
# 18	18	Belgium Thierry Boutsen	Arrows-BMW	1:32.530	1:25.115	+3.412
# 19	2	West Germany Manfred Winkelhock	Brabham-BMW	no time	1:25.289	+3.586
# 20	26	Italy Andrea de Cesaris	Ligier-Renault	1:33.398	1:26.082	+4.379
# 21	25	France François Hesnault	Ligier-Renault	1:34.233	1:26.701	+4.998
# 22	24	Italy Piercarlo Ghinzani	Osella-Alfa Romeo	1:31.336	1:26.840	+5.137
# 23	14	Austria Gerhard Berger	ATS-BMW	1:44.966	1:28.106	+6.403
# 24	30	Austria Jo Gartner	Osella-Alfa Romeo	1:33.540	1:28.229	+6.526
# 25	21	Italy Mauro Baldi	Spirit-Hart	1:36.483	1:29.001	+7.298
# 26	10	United Kingdom Jonathan Palmer	RAM-Hart	1:40.344	1:29.397	+7.694
# 27	9	France Philippe Alliot	RAM-Hart	1:34.839	1:30.406	+8.703
#
#     """
#
#     My_Ergast().insert_qualy_data(1984, 16, data)

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
