#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:36:38 2017

@author: dominikruchti
"""

import numpy as np
from scipy.spatial import distance
from scipy.stats import multivariate_normal
import copy
from matplotlib import pyplot as plt

plt.close('all')


# Generic synthetic demand
class SyntheticDemand:

    def __init__(self, K, O, D, G):

        self.Lag = K
        self.Nr_origins = O
        self.Nr_destionations = D

        self.Grid_dim = G

        g = G / 100.

        x, y = np.mgrid[0:G:g, 0:G:g]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x;
        pos[:, :, 1] = y

        self.Grid = pos

        self.coordinates = [(self.Grid[k, r])
                            for k in range(0, len(self.Grid))
                            for r in xrange(self.Grid.shape[0])]

        self.create_origins()
        self.create_destinations()

    def create_origins(self):
        self.Origins = []

        # np.random.seed(self.Nr_origins)
        np.random.seed(None)
        for k in range(self.Nr_origins):
            x = np.random.randint(0, self.Grid_dim)
            y = np.random.randint(0, self.Grid_dim)
            self.Origins.append(np.array([x, y]))

    def create_destinations(self):
        self.Destinations = []

        # np.random.seed(self.Nr_destionations)
        np.random.seed(None)
        for k in range(self.Nr_destionations):
            x = np.random.randint(0, self.Grid_dim)
            y = np.random.randint(0, self.Grid_dim)
            self.Destinations.append(np.array([x, y]))

    def create_prob_distribution(self, c, plt_opt):
        if c == "O":
            centres = copy.copy(self.Origins)
        elif c == "D":
            centres = copy.copy(self.Destinations)
        else:
            print "Wrong argument!"

        pdf = []
        pdf_tot = np.zeros((self.Grid[:, :, 0].shape[0], self.Grid[:, :, 1].shape[1]))

        for k in range(len(centres)):
            mean = centres[k]
            v = np.random.randint(1, 4)
            var = v * self.Grid_dim / len(centres)
            pdf.append(multivariate_normal(mean, var))

            pdf_tot += pdf[k].pdf(self.Grid)

        # Nomralize
        pdf_tot += 0.0005
        pdf_cum = pdf_tot / (sum(sum(pdf_tot)))

        if c == "O":
            self.Trip_Origin_PDF = pdf_cum
        elif c == "D":
            self.Trip_Destination_PDF = pdf_cum

        if plt_opt:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            if c == "O":
                ax.plot_surface(self.Grid[:, :, 0], self.Grid[:, :, 1], pdf_cum, cmap='viridis',
                                linewidth=0)
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                plt.show()

                plt.figure()
                plt.contourf(self.Grid[:, :, 0], self.Grid[:, :, 1], pdf_cum)
                plt.title('Trip Origin Distribution')
            elif c == "D":
                ax.plot_surface(self.Grid[:, :, 0], self.Grid[:, :, 1], -1 * pdf_cum,
                                cmap='viridis', linewidth=0)
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                plt.show()

                plt.figure()
                plt.contourf(self.Grid[:, :, 0], self.Grid[:, :, 1], pdf_cum)
                plt.title('Trip Destination Distribution')

    def draw_samples(self, PDF, nr_samples, plt_opt):

        if len(PDF) != len(self.coordinates):
            # prob = [PDF[x,y] for x in xrange(PDF.shape[0]) for y in xrange(PDF.shape[1])]
            prob = np.reshape(PDF, (-1, 1))[:, 0]

        else:
            prob = PDF
            PDF = np.reshape(prob, (np.int(np.sqrt(len(prob))), np.int(np.sqrt(len(prob)))))

        sample_indices = np.random.choice(np.arange(len(prob)), nr_samples, p=prob)
        samples = [self.coordinates[k] for k in sample_indices]
        samples_x = [samples[k][0] for k in xrange(len(samples))]
        samples_y = [samples[k][1] for k in xrange(len(samples))]

        if plt_opt:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(self.Grid[:, :, 0], self.Grid[:, :, 1], PDF, cmap='viridis',
                            linewidth=0)
            ax.scatter(samples_x, samples_y)

            plt.figure()
            plt.contourf(self.Grid[:, :, 0], self.Grid[:, :, 1], PDF)
            plt.plot(samples_x, samples_y, 'o')

        return samples, samples_x, samples_y

    def generate_trips(self, samples, plt_opt):

        Dest_PDF = copy.copy(self.Trip_Destination_PDF)
        # prob = [Dest_PDF[x,y] for x in xrange(Dest_PDF.shape[0]) for y in xrange(Dest_PDF.shape[1])]
        prob = np.reshape(Dest_PDF, (-1, 1))[:, 0]
        # ind = []
        Trips = []

        for k in range(0, len(samples)):
            # print "Sampling Trip %d" %k
            ind = [distance.euclidean(samples[k], j) >= self.Lag * 20 for j in self.coordinates]
            # ind.append([abs(samples[k][0]-j[0]+samples[k][1]-j[1])<=self.Lag*20 for j in self.coordinates])
            prob_temp = copy.copy(prob)
            for index, item in enumerate(ind):
                if item:
                    prob_temp[index] = 0

            pdf_cum = prob_temp / (sum(prob_temp))

            D_samples, D_samples_x, D_samples_y = self.draw_samples(pdf_cum, 1, 0)

            #            plt.figure()
            #            plt.subplot(2,1,1)
            #            plt.contourf(self.Grid[:,:,0], self.Grid[:,:,1], Dest_PDF)
            #            plt.subplot(2,1,2)
            #            plt.contourf(self.Grid[:,:,0], self.Grid[:,:,1], np.reshape(pdf_cum,(np.sqrt(len(pdf_cum)),np.sqrt(len(pdf_cum)))))
            #            plt.plot(D_samples_x[-1], D_samples_y[-1],'o',c='black')
            #            plt.plot(samples[k][0], samples[k][1],'o',c='red')

            Trips.append((samples[k], D_samples[-1]))

        if plt_opt:
            plt.figure()
            plt.contourf(self.Grid[:, :, 0], self.Grid[:, :, 1], Dest_PDF)
            for k in xrange(len(Trips)):
                x = Trips[k][0][0]
                y = Trips[k][0][1]
                dx = Trips[k][1][0] - x
                dy = Trips[k][1][1] - y

                plt.arrow(x, y, dx, dy, width=0.1, head_width=2)
                plt.plot(x, y, 'o', c='black')
        return Trips


# Generic synthetic grid network
class SyntheticNetwork:
    def __init__(self, N, V, Cs, Cv, B, T, W, K, vehicle_speed):

        self.nr_regions = N
        self.grid_space = np.int(np.sqrt(N))
        self.nr_vehicles = V

        self.C_s = Cs
        self.C_v = Cv

        self.nr_bikes = B
        self.time_horizon = T
        self.time_window = W
        self.nr_lag_steps = K

        self.centres = None
        self.adjacency_matrix = None
        self.F = None

        self.create_centres()
        self.dist = self.distance_matrix()
        self.create_adjacency_matrix(vehicle_speed)
        self.ds_0 = self.create_initial_bike_distr()
        self.dv_0 = self.create_initial_vehicle_load()
        self.z_0 = self.create_initial_vehicle_distr()

        self.create_demand()

    def __str__(self):
        s = "\n"
        s += "Network has %d regions, %d vehicles, and %d bikes.\n" % (self.nr_regions,
                                                                       self.nr_vehicles,
                                                                       self.nr_bikes)
        s += "Time horizon is %d steps, and maximum lag is %d steps.\n" % (self.time_horizon,
                                                                           self.nr_lag_steps)
        s += "Regions hold %.1f bikes, and vehicles hold %.1f bikes on average." % \
             (np.mean(self.C_s), np.mean(self.C_v))
        return s

    def print_stats(self):
        print "Initial vehicle conditions:"
        N, V = self.nr_regions, self.nr_vehicles
        for v in range(V):
            z_temp = self.z_0[v:(N*V)+v:V, 0].reshape(N)
            start_location = np.argwhere(z_temp)[0]
            print "  Vehicle #%2d starts in region %2d with %2d/%2d slots filled." % \
                  (v, start_location[0], self.dv_0[v], self.C_v[v])
        print "Initial bike numbers by region:"
        print " ", [int(x) for x in self.ds_0.squeeze()]
        if np.sum(self.ds_0) != self.nr_bikes:
            print "Warning: %d (and not %d) bikes in network!" % (np.sum(self.ds_0),
                                                                  self.nr_bikes)
        # raw_input('')

    def create_centres(self):

        self.centres = []
        for k in range(0, self.grid_space):
            for j in range(0, self.grid_space):
                x = j * 100. / self.grid_space + 100. / (2 * self.grid_space)
                y = k * 100. / self.grid_space + 100. / (2 * self.grid_space)
                self.centres.append(np.array([x, y]))

    def create_adjacency_matrix(self, max_travel_dist):
        N = self.nr_regions
        self.adjacency_matrix = np.zeros((N, N), dtype=int)
        ind = np.where(self.dist <= max_travel_dist)
        self.adjacency_matrix[ind] = 1
        # print "Adjacency matrix:"
        # print self.adjacency_matrix

    def create_initial_bike_distr(self):

        # np.random.seed(self.nr_bikes)
        N = self.nr_regions
        B = np.float(self.nr_bikes)

        r = np.random.randint(0, B / 2, size=(N, 1))
        x_0 = np.round((r * B / sum(r)))
        for k in range(N):
            x_0[k] = min(x_0[k], self.C_s[k])

        while int(sum(x_0)) > self.nr_bikes:
            i = np.random.randint(0, N)
            if x_0[i] > 0:
                x_0[i] -= 1
        while int(sum(x_0)) < self.nr_bikes:
            i = np.random.randint(0, N)
            if x_0[i] < self.C_s[i]:
                x_0[i] += 1

        return x_0

    def create_initial_vehicle_distr(self):

        N = self.nr_regions
        V = self.nr_vehicles

        z_0 = np.zeros([N * N * V, 1])

        available_regions = np.arange(N)

        initial_regions = []
        for v in range(0, V):
            # select random index out of available regions
            r = np.random.randint(0, len(available_regions), size=1)
            initial_regions.append(np.int(available_regions[r]))
            # available_regions = np.delete(available_regions,r)

            ir = initial_regions[v]
            z_0[ir * V + v] = 1

        return z_0

    def create_initial_vehicle_load(self):

        # np.random.seed(self.nr_vehicles)

        x_0 = np.zeros((self.nr_vehicles, 1))

        return x_0

    def modify_nr_vehicles(self, V, C_v):
        self.nr_vehicles = V
        self.C_v = C_v
        self.z_0 = self.create_initial_vehicle_distr()
        self.dv_0 = self.create_initial_vehicle_load()

    def create_demand(self):

        G = 100.
        Nr_Origins = 3
        Nr_Destinations = 5

        Demand = SyntheticDemand(self.nr_lag_steps, Nr_Origins, Nr_Destinations, G)

        F = [[np.zeros((self.nr_regions, self.nr_regions))
              for k in range(self.nr_lag_steps + 1)]
             for t in range(self.time_horizon + 1)]

        total_demand = 0

        for k in range(0, self.time_window):

            Demand.create_origins()
            Demand.create_destinations()

            for t in range(0, self.time_horizon / self.time_window):
                Demand.create_prob_distribution("O", 0)
                Demand.create_prob_distribution("D", 0)

                n = max(np.int(np.random.normal(0.15 * self.nr_bikes, 0.075 * self.nr_bikes)), 0)
                origin_samples, ox, oy = Demand.draw_samples(Demand.Trip_Origin_PDF, n, 0)

                Trips = Demand.generate_trips(origin_samples, 0)

                # F.append([np.zeros((self.nr_regions,self.nr_regions))]*(self.Lag+1))

                for j in range(0, len(Trips)):
                    Lag = np.int(distance.euclidean(Trips[j][0], Trips[j][1]) / 20)
                    o = np.int(Trips[j][0][0] / (G / np.sqrt(self.nr_regions))) + np.int(
                        np.sqrt(self.nr_regions) * np.int(
                            Trips[j][0][1] / (G / np.sqrt(self.nr_regions))))
                    d = np.int(Trips[j][1][0] / (G / np.sqrt(self.nr_regions))) + np.int(
                        np.sqrt(self.nr_regions) * np.int(
                            Trips[j][1][1] / (G / np.sqrt(self.nr_regions))))
                    # print F[-1][Lag][o,d]
                    try:
                        F[k * self.time_horizon / self.time_window + t + 1][Lag][o, d] += 1
                    except:
                        print "Error"

                total_demand += len(Trips)
                print "  %d trips created at time %d." % \
                      (len(Trips), k * self.time_horizon / self.time_window + t + 1)

        print "  Total demand is %d trips; %.2f trips/bike." % (total_demand,
                                                                float(total_demand) / self.nr_bikes)

        self.F = F

    def distance_matrix(self):

        dist_matrix = np.zeros((self.nr_regions, self.nr_regions))

        for k in range(self.nr_regions):
            for l in range(self.nr_regions):
                dist_matrix[k, l] = (distance.euclidean(self.centres[k], self.centres[l]))

        return dist_matrix

    def repos_penalty(self, t, P):
        dist = self.dist
        for i in range(0, dist.shape[0]):
            dist[i, i] = 0

        return P * dist

    def trip_reward(self, t, k, R1):

        dist = self.dist
        for i in range(0, dist.shape[0]):
            dist[i, i] = R1

        return dist
