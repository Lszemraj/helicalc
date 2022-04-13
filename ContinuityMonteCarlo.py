import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from funcs_file import *
import dash
from dash import Dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pickle as pkl

df_dict = load_all_geoms(return_dict=True)
generate_dict, conductor_dict = make_conductor_dicts(df_dict)


number_conductors = len(conductor_dict.keys())


def check_continuity(cond_in, cond_out, df_dict, conductor_dict, cutoff):
    N = 1000
    key_out = conductor_dict[cond_out]
    key_in = conductor_dict[cond_in]
    xs_out, ys_out, zs_out = generate_dict[key_out](df_dict[key_out], cond_out, conductor_end='out', N=N, N_rel=False)
    xs_in, ys_in, zs_in = generate_dict[key_in](df_dict[key_in], cond_in, conductor_end='in', N=N, N_rel=False)

    x_out_matrix = np.repeat(xs_out.reshape((-1, 1)), N, axis=1)
    x_in_matrix = np.repeat(xs_in.reshape((-1, 1)), N, axis=1).T
    y_out_matrix = np.repeat(ys_out.reshape((-1, 1)), N, axis=1)
    y_in_matrix = np.repeat(ys_in.reshape((-1, 1)), N, axis=1).T
    z_out_matrix = np.repeat(zs_out.reshape((-1, 1)), N, axis=1)
    z_in_matrix = np.repeat(zs_in.reshape((-1, 1)), N, axis=1).T
    #calculate distances between points
    dist_matrix = ((x_out_matrix - x_in_matrix) ** 2 + (y_out_matrix - y_in_matrix) ** 2 + (
                z_out_matrix - z_in_matrix) ** 2) ** (1 / 2)
    min_dist = np.min(dist_matrix)
    if min_dist < cutoff:
        connected = True
        #print(f'Conductor {cond_out} (out) is connected to conductor {cond_in} (in)')
    else:
        connected = False
        #print(f'Conductor {cond_out} (out) is NOT connected to conductor {cond_in} (in)')

    return connected

connected_matrix = np.zeros((len(conductor_dict), len(conductor_dict)), dtype = bool)
close_matrix = np.zeros((len(conductor_dict), len(conductor_dict)), dtype = bool)

list_in_order = []

#loop through and check continuity

sorted_dict = sorted(conductor_dict.items())

for i,item in enumerate(sorted_dict):
    #if i == 0:

        N0, C0 = item  # N0 is arc number, C0 is type of conductor
        list_in_order.append(N0)
        for j, item2, in enumerate(sorted_dict):  # this does same thing
            cn, ctype = item2  # cn is conductor number, ctype is the type of conductor for the secondary conductor we are checking
            if cn != N0:  # make sure we are not checking if the same conductor to itself
                x = check_continuity(cn, N0, df_dict, conductor_dict, 1e-4)  # uses the matrix stuff
                if x:
                    connected = True
                    connected_matrix[i, j] = connected
                    list_in_order.append(cn)
                else:
                    connected = False
                    connected_matrix[i, j] = connected
                y = check_continuity(cn, N0, df_dict, conductor_dict, 5e-3)  # checks if "close"
                if y:
                    close = True
                    close_matrix[i, j] = close
                else:
                    close = False
                    close_matrix[i, j] = close
'''
    #else:
        N0 = list_in_order[-1]
        C0 = conductor_dict[N0]
        for j, item2, in enumerate(conductor_dict.items()):  # this does same thing
            cn, ctype = item2  # cn is conductor number, ctype is the type of conductor for the secondary conductor we are checking
            if cn != N0:  # make sure we are not checking if the same conductor to itself
                x = check_continuity(cn, N0, df_dict, conductor_dict, 1e-4)  # uses the matrix stuff
                if x:
                    connected = True
                    connected_matrix[i, j] = connected
                    list_in_order.append(cn)
                else:
                    connected = False
                    connected_matrix[i, j] = connected
                    
                y = check_continuity(cn, N0, df_dict, conductor_dict, 5e-3)  # checks if "close"
                if y:
                    close = True
                    close_matrix[i, j] = close
                else:
                    close = False
                    close_matrix[i, j] = close
'''
#print(list_in_order)



with open("close_matrix4.pkl", 'wb') as file:
    pkl.dump(close_matrix, file)

with open("connected_matrix3=4.pkl", 'wb') as file:
    pkl.dump(connected_matrix, file)

with open("lis_in_order4.pkl", 'wb') as file:
    pkl.dump(list_in_order, file)





'''       
    #i is index, item is a list of the arc # and then type of arc
    for i, item in enumerate(conductor_dict.items()):
    N0, C0 = item #N0 is arc number, C0 is type of conductor
    for j, item2, in enumerate(conductor_dict.items()): #this does same thing
        cn, ctype = item2 #cn is conductor number, ctype is the type of conductor for the secondary conductor we are checking
        if cn != N0: #make sure we are not checking if the same conductor to itself
            x = check_continuity(cn, N0, df_dict, conductor_dict, 1e-4) #uses the matrix stuff
            if x:
                connected = True
                connected_matrix[i, j] = connected

            else:
                connected = False
                connected_matrix[i, j] = connected
            y =check_continuity(cn, N0, df_dict, conductor_dict, 5e-3) #checks if "close"
            if y:
                close =True
                close_matrix[i,j] = close
            else:
                close = False
                close_matrix[i,j] = close


print(connected_matrix)
'''