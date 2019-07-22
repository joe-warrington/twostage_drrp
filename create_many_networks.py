import numpy as np
import os
import pickle
from drrp import SyntheticNetwork

"""Create a sequence of random network instances and store them in folders in network_data/"""
n_v_array = {9: [1, 3], 16: [1, 5], 25: [1, 5, 9], 36: [1, 5, 11],
             64: [1, 9, 15], 100: [1, 9, 19], 225: [1, 13, 25], 400: [1, 15, 35]}
for n in [3, 4, 5, 6, 8, 10, 15, 20, 30]:  # Number of nodes n along the edge of the square grid
    for T in [12]:  # Number of time steps T in planning horizon
        for i in range(1, 11):  # Controls number of instances i and their labels
            # Number of nodes
            N = n * n
            # Total number of bikes in system
            B = N * 5
            # Planning horizon
            W = 6
            # Time Horizon for trips
            K = 2
            # Station capacity
            C_s = np.ceil(2. * B / N) * np.ones(N)
            # Vehicle count and capacity
            V = 1
            vehicle_cap = 5
            C_v = vehicle_cap * np.ones(V)
            # Vehicle speed
            vehicle_speed = 100. / n * 1.25

            # Name and create target directory
            directory = "network_data/"
            network_folder = directory + "N%03d_V%02d_T%02d" % (N, V, T)
            if not os.path.exists(network_folder):
                os.makedirs(network_folder)
            filename = network_folder + '/instance_%02d' % i
            print "Creating network: N=%d, V=%d, T=%d, instance %d" % (N, V, T, i)

            # Generate network object
            nw = SyntheticNetwork(N, V, C_s, C_v, B, T, W, K, vehicle_speed)
            with open(filename + '.pkl', 'wb') as output:
                pickle.dump(nw, output, pickle.HIGHEST_PROTOCOL)

            for V in n_v_array[N]:
                # Update network object for new V (don't generate new customer demand matrices)
                nw.modify_nr_vehicles(V, vehicle_cap * np.ones(V))
                # Name and create target directory
                directory = "network_data/"
                network_folder = directory + "N%03d_V%02d_T%02d" % (N, V, T)
                if not os.path.exists(network_folder):
                    os.makedirs(network_folder)
                filename = network_folder + '/instance_%02d' % i
                print "Creating network: N=%d, V=%d, T=%d, instance %d" % (N, V, T, i)

                # Save network object modified for new value of V
                with open(filename + '.pkl', 'wb') as output:
                    pickle.dump(nw, output, pickle.HIGHEST_PROTOCOL)
