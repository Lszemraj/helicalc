import numpy as np
import pickle as pkl
from funcs_file import make_conductor_dicts, load_all_geoms

df_dict = load_all_geoms()
generate_dict, conductor_dict, id_column_dict = make_conductor_dicts(df_dict)


# these tell us which index corresponds to which conductor key
sorted_dict = sorted(conductor_dict.items())
#print(sorted_dict)
index_dict = {i:k[0] for i,k in enumerate(sorted_dict)} # save this
#print(index_dict) #First num is index in matrix, second num is conductor index


datadir = '/home/shared_data/helicalc_params/'
min_dist_file = datadir+'min_dist_matrix.pkl'
connected_file = datadir+'connected_matrix.pkl'
close_file = datadir+'close_matrix.pkl'

with open(min_dist_file, 'rb') as file:
    min_dist_matrix = pkl.load(file)
# correct the 0 along the diagonal

min_dist_matrix = min_dist_matrix + 1000*np.identity(len(min_dist_matrix))

with open(connected_file, 'rb') as file:
    connected_matrix = pkl.load(file)
    connected_array = np.array(connected_matrix)

with open(close_file, 'rb') as file:
    close_matrix = pkl.load(file)



connection_dict = {}

def find_neighbour(index_dict, min_dist_matrix, connected_array):
    for key, value in enumerate(index_dict):
        index_number = key
        conductor_num = value
        try:
            x = int(np.argwhere(connected_matrix[index_number]))
            connected_conductor = index_dict[x]
            connection_dict[value] = connected_conductor
        except:
            closest = np.argwhere(min_dist_matrix[key] == np.amin(min_dist_matrix[key]) )
            closest_conductor = index_dict[int(closest)]
            connection_dict[value] = closest_conductor


reverse_index_dict = {k[0]:i for i,k in enumerate(sorted_dict)}

def find_correct_order(reverse_index_dict, index_dict, close_matrix):
    conductor_chain = [index_dict[0]]
    conductors_checked = [index_dict[0]]
    conductor_list_chain = [] #store diff chains
    for cond_num in range(0, len(close_matrix)): #81 total conductors
        try:
            initial_cond = reverse_index_dict[conductor_chain[-1]]
            closest = np.argwhere(close_matrix[initial_cond] == True)
            closest_conductor = index_dict[int(closest)]
            conductor_chain.append(closest_conductor)
            conductors_checked.append(closest_conductor)
        except:
            try:
                last_cond = reverse_index_dict[conductor_chain[0]]
                closest = np.argwhere(close_matrix[:, last_cond] == True)
                closest_conductor = index_dict[int(closest)]
                conductor_chain.insert(0, closest_conductor)
                conductors_checked.append(closest_conductor)
            except: #chain is finished
                conductor_list_chain.append(conductor_chain)
                for i in reverse_index_dict.keys():
                    if i not in conductors_checked:
                        conductor_chain = [i]
                        conductors_checked.append(i)
                        break
    conductor_list_chain.append(conductor_chain)
    return conductor_list_chain, conductors_checked






x, y = find_correct_order(reverse_index_dict, index_dict, close_matrix)
print("conductor list chain:", x)
print("conductors checked:", y)
print("conductors not in conductors_checked:")
for i in reverse_index_dict.keys():
    if i not in y:
        print(i)
