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

df_connected = pd.read_pickle("/home/lillie/PycharmProjects/pythonProject/Current-Continuity/connected_matrix3=4.pkl")
df_close = pd.read_pickle("/home/lillie/PycharmProjects/pythonProject/Current-Continuity/close_matrix4.pkl")
df = load_all_geoms(return_dict=True)
a = len(df['straights'])
b = len(df['arcs'])
c = len(df['arcs_transfer'])
length = a +b +c
print("Number of conductors:", length)
#print(df_connected)
output_input = np.argwhere(df_connected)
print(output_input)
connections = np.sum(df_connected, axis = 0)
close = np.sum(df_close, axis = 0)
print("Connection Matrix", connections)
print("Close Matrix", close)
#print(output_input)
unconnected_coils = []
unconnected_coils_output_input = []

for i in range(0, len(connections)):
    if connections[i] == 0:
        #unconnected_coils_output_input.append(output_input[i])
        unconnected_coils.append(i+1)

print(unconnected_coils)
print("Number of unconnected conductors:", len(unconnected_coils))