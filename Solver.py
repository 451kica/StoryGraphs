#!/usr/bin/env python
# coding: utf-8



import networkx as nx
import random
import numpy as np
from numpy import array
import cvxpy as cp
import cvxopt

np.set_printoptions(precision=3)


def paths(tree, cur=()):
    if not tree:
        yield cur
    else:
        for n, s in tree.items():
            for path in paths(s, cur + (n,)):
                yield path


def fill_tree(tree, n, previous, number_of_segments, edges):
    if n == number_of_segments + 1:
        return tree

    for i in range(2):
        if previous[1] == edges[n * 2 + i][0]:  # check adjacency
            tree[edges[n * 2 + i]] = {}
            fill_tree(tree[edges[n * 2 + i]], n=n + 1, previous=edges[n * 2 + i],
                      number_of_segments=number_of_segments, edges=edges)



random.seed(1)
class Solver:

    def __init__(self, L=1.0, number_of_segments=None):
        self.L = L
        self.number_of_segments = number_of_segments

    @staticmethod
    def make_graph_4segments(list_len_segments):
        G = nx.Graph()
        for j in range(1, 1 + list_len_segments[0]):
            G.add_edge(0, j, weight=0)
        for i in range(1, 1 + list_len_segments[0]):
            for j in range(1 + list_len_segments[0], 1 + list_len_segments[0] + list_len_segments[1]):
                G.add_edge(i, j, weight=random.uniform(0, 1))
        for i in range(1 + list_len_segments[0], 1 + list_len_segments[0] + list_len_segments[1]):
            for j in range(1 + list_len_segments[0] + list_len_segments[1],
                           1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2]):
                G.add_edge(i, j, weight=random.uniform(0, 1))
        for i in range(1 + list_len_segments[0] + list_len_segments[1],
                       1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2]):
            for j in range(1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2],
                           1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2] + list_len_segments[3]):
                G.add_edge(i, j, weight=random.uniform(0, 1))
        for i in range(1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2],
                       1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2] + list_len_segments[3]):
            G.add_edge(i, 1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2] + list_len_segments[3], weight=0)
        return G

    @staticmethod
    def make_graph_3segments(list_len_segments):
        G = nx.Graph()
        for j in range(1, 1 + list_len_segments[0]):
            G.add_edge(0, j, weight=0)
        for i in range(1, 1 + list_len_segments[0]):
            for j in range(1 + list_len_segments[0], 1 + list_len_segments[0] + list_len_segments[1]):
                G.add_edge(i, j, weight=random.uniform(0, 1))
        for i in range(1 + list_len_segments[0], 1 + list_len_segments[0] + list_len_segments[1]):
            for j in range(1 + list_len_segments[0] + list_len_segments[1],
                           1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2]):
                G.add_edge(i, j, weight=random.uniform(0, 1))
        for i in range(1 + list_len_segments[0] + list_len_segments[1],
                       1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2]):
            G.add_edge(i, 1 + list_len_segments[0] + list_len_segments[1] + list_len_segments[2], weight=0)
        return G


    def Q_matrix(self, G, L, edges_list):
        Q = np.zeros((len(edges_list), len(edges_list)))
        for i in range(len(edges_list)):
            for j in range(len(edges_list)):
                if edges_list[i][1] == edges_list[j][0]:
                    Q[i][j] = L * (1 / 8) * ((G[edges_list[j][0]][edges_list[j][1]]['weight'] -
                                              G[edges_list[i][0]][edges_list[i][1]]['weight']) ** 2)
                    Q[j][i] = L * (1 / 8) * ((G[edges_list[j][0]][edges_list[j][1]]['weight'] -
                                              G[edges_list[i][0]][edges_list[i][1]]['weight']) ** 2)
                Q[i][i] = 0
        return Q

    def part_of_c_vector(self, G, L, edges_list):
        c = np.zeros((len(edges_list), 1))
        for i in range(len(edges_list)):
            for j in range(len(edges_list)):
                if edges_list[i][1] == edges_list[j][0]:
                    c[i] = c[i] + L * (1 / 4) * ((G[edges_list[j][0]][edges_list[j][1]]['weight'] -
                                                  G[edges_list[i][0]][edges_list[i][1]]['weight']) ** 2)
        return c

    def Av_matrix(self, v, edges_list, edge_segments, number_of_segments):  # make test matrix A: numb of nonzeros =2* in and out connections of node
        av = np.zeros((len(edges_list), 1))
        for i in range(number_of_segments + 1):
            for j in range(len(edge_segments[i])):
                if v == edge_segments[i][j][0]:
                    av[edges_list.index(edge_segments[i][j])] = 1
                if v == edge_segments[i][j][1]:
                    av[edges_list.index(edge_segments[i][j])] = -1

        Av = np.zeros((len(edges_list), len(edges_list)))

        Av = np.hstack([Av, (1 / 2) * av])
        new_vector = np.vstack([(1 / 2) * av, 0]).T
        Av = np.vstack([Av, new_vector])

        # A_matrices[l]=  Av
        # l +=1
        return Av

    def solution_cost_function_value(self,G,edges_path):
        v = [G[edges_path[0][0]][edges_path[0][1]]['weight'],
             G[edges_path[1][0]][edges_path[1][1]]['weight'],
             G[edges_path[2][0]][edges_path[2][1]]['weight'],
             G[edges_path[3][0]][edges_path[3][1]]['weight']]

        if self.number_of_segments == 4:
            v.append(G[edges_path[4][0]][edges_path[4][1]]['weight'])

        return Solver.cost_function(v, self.L)

    def solve(self, G, list_len_segments):

        self.number_of_segments = len(list_len_segments)
        number_of_nodes_with_start_and_destination = 2 + np.sum(list_len_segments)

        edges_list = list(G.edges)

        Q = self.Q_matrix(G, self.L, edges_list)

        edges_weights = np.zeros((len(edges_list), 1))
        for i in range(len(edges_list)):
            edges_weights[i] = G[edges_list[i][0]][edges_list[i][1]]['weight']

        c = self.part_of_c_vector(G, self.L, edges_list) + (1 / 2) * edges_weights

        W = np.hstack([Q, c])
        new_row = np.vstack([c, 0]).T
        W = np.vstack([W, new_row])

        edge_segments = []
        edge_segments.extend([edges_list[0:list_len_segments[0]]])
        edge_segments.extend([edges_list[list_len_segments[0]:(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]))]])
        if len(list_len_segments) == 4:
            edge_segments.extend([edges_list[(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):(
                        (list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])) + (
                            list_len_segments[1] * list_len_segments[2]))]])
            edge_segments.extend([edges_list[((list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])) + (
                        list_len_segments[1] * list_len_segments[2])):((list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])) + (
                        list_len_segments[1] * list_len_segments[2]) + (list_len_segments[2] * list_len_segments[3]))]])
            edge_segments.extend([edges_list[(len(edges_list) - list_len_segments[3]):len(edges_list)]])
        elif  len(list_len_segments) == 3:
            edge_segments.extend([edges_list[(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):(
                        (list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])) + (
                            list_len_segments[1] * list_len_segments[2]))]])
            edge_segments.extend([edges_list[(len(edges_list) - list_len_segments[2]):len(edges_list)]])

        asource = np.zeros((len(edges_list), 1))
        for i in range(list_len_segments[0]):
            asource[i] = 1

        A_source = np.zeros((len(edges_list), len(edges_list)))
        A_source = np.hstack([A_source, (1 / 2) * asource])
        new_vector = np.vstack([(1 / 2) * asource, 0]).T
        A_source = np.vstack([A_source, new_vector])


        adestination = np.zeros((len(edges_list), 1))

        for i in range(list_len_segments[-1]):
            adestination[(len(edges_list) - 1) - i] = 1

        A_destination = np.zeros((len(edges_list), len(edges_list)))
        A_destination = np.hstack([A_destination, (1 / 2) * adestination])
        new_vector = np.vstack([(1 / 2) * adestination, 0]).T
        A_destination = np.vstack([A_destination, new_vector])

        A_matrices = {}
        l = 1

        # for i in range(1,7): #instead 5 , number of nodes*number of segments - 1
        for i in range(1,
                       (number_of_nodes_with_start_and_destination) - 1):  # instead 5 , number of nodes*number of segments - 1
            Av = self.Av_matrix(v=i, edges_list=edges_list, edge_segments=edge_segments, number_of_segments=self.number_of_segments)
            A_matrices[l] = Av
            l += 1
        A_matrices[0] = A_source
        A_matrices[(number_of_nodes_with_start_and_destination - 1)] = A_destination

        cv = np.zeros((len(A_matrices)))
        for i in range(1, list_len_segments[0] + 1):
            cv[i] = 1 - list_len_segments[1]
        for i in range(1, list_len_segments[1] + 1):
            cv[list_len_segments[0] + i] = list_len_segments[0] - list_len_segments[2]
        if len(list_len_segments) == 4:
            for i in range(1, list_len_segments[2] + 1):
                cv[list_len_segments[0] + list_len_segments[1] + i] = list_len_segments[1] - list_len_segments[3]
            for i in range(1, list_len_segments[3] + 1):
                cv[list_len_segments[0] + list_len_segments[1] + list_len_segments[2] + i] = list_len_segments[2] - 1
        elif len(list_len_segments)==3:
            for i in range(1, list_len_segments[2] + 1):
                cv[list_len_segments[0] + list_len_segments[1] + i] = list_len_segments[1] - 1
        cv[0] = 2 - list_len_segments[0]
        cv[-1] = 2 - list_len_segments[-1]



        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        Z = cp.Variable((len(edges_list) + 1, len(edges_list) + 1), PSD=True)  # PSD=True
        # The operator >> denotes matrix inequality.

        constraints = [Z >> 0]  # ,cp.trace(Z) ==(len(edges_list)+1)
        constraints += [Z[j][j] == 1 for j in range(len(edges_list) + 1)]
        constraints += [
            # cp.trace(A_matrices[0]@Z) == cv[0]
            cp.trace(A_matrices[i] @ Z) == cv[i] for i in range(len(A_matrices))
        ]

        prob = cp.Problem(cp.Minimize(cp.trace(W @ Z)),
                          constraints)
        prob.solve(solver='CVXOPT')

        # Print result.
        print("The optimal value is", prob.value)
        #print("A solution Z is")
        #print(Z.value)

        z = np.array(Z.value[((Z.value).shape[0] - 1)])
        print("The z vector is", z)


        if len(list_len_segments) == 4:
            sdp_solution = self.sdp_solution_4segments(edges_weights, edges_list, list_len_segments, z)
            rounding2_solution_cost_function, rounding2_solution = self.get_solution_rounding_2_4segments(G, z, list_len_segments)
            brute_force_solution, brute_force_positions = self.brute_force_solution_4segments(G=G,
                                                                                               list_len_segments=list_len_segments,
                                                                                               edges_list=edges_list)
           # shortest_path_4segments(self, G, list_len_segments, edges_list)
        elif len(list_len_segments) == 3:
            sdp_solution = self.sdp_solution_3segments(edges_weights, edges_list, list_len_segments, z)


            rounding2_solution_cost_function, rounding2_solution = self.get_solution_rounding_2_3segments(G, z,
                                                                                                           list_len_segments)

            brute_force_solution, brute_force_positions = self.brute_force_solution_3segments(G=G, list_len_segments=list_len_segments,
                                                             edges_list=edges_list)
           # shortest_path_3segments(self, G, list_len_segments, edges_list)

        return sdp_solution, brute_force_solution, brute_force_positions, W, rounding2_solution

    def sdp_solution_3segments(self, edges_weights, edges_list, list_len_segments, z):
        sdp_solution = np.empty((len(z)))
        sdp_solution[:] = np.nan
        for i in range(0, list_len_segments[0]):
            if z[i] == np.max(z[0:list_len_segments[0]]):
                sdp_solution[i] = edges_weights[i]
        for i in range(list_len_segments[0], list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):
            if z[i] == np.max(z[list_len_segments[0]:list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])]):
                sdp_solution[i] = edges_weights[i]
        for i in range(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]), list_len_segments[0] + (
                list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2])):
            if z[i] == np.max(z[list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]):list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2])]):
                sdp_solution[i] = edges_weights[i]
        for i in range(
                list_len_segments[0] + (list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]),
                len(edges_list)):
            if z[i] == np.max(z[list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]):len(edges_list)]):
                sdp_solution[i] = edges_weights[i]
        return sdp_solution

    def sdp_solution_4segments(self, edges_weights, edges_list, list_len_segments,z ):
        sdp_solution = np.empty((len(z)))
        sdp_solution[:] = np.nan
        for i in range(0, list_len_segments[0]):
            if z[i] == np.max(z[0:list_len_segments[0]]):
                sdp_solution[i] = edges_weights[i]
        for i in range(list_len_segments[0], list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):
            if z[i] == np.max(
                    z[list_len_segments[0]:list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])]):
                sdp_solution[i] = edges_weights[i]
        for i in range(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]),
                       list_len_segments[0] + (
                               list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[
                           2])):
            if z[i] == np.max(
                    z[list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]):list_len_segments[0] + (
                            list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[
                        2])]):
                sdp_solution[i] = edges_weights[i]
        for i in range(
                list_len_segments[0] + (
                        list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]),
                list_len_segments[0] + (
                        list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2] +
                        list_len_segments[2] * list_len_segments[3])):
            if z[i] == np.max(z[list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]):
            list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2] +
                    list_len_segments[2] * list_len_segments[3])]):
                sdp_solution[i] = edges_weights[i]
        for i in range(list_len_segments[0] + (
                list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2] +
                list_len_segments[2] * list_len_segments[3]),
                       len(edges_list)):
            if z[i] == np.max(z[list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2] +
                    list_len_segments[2] * list_len_segments[3]):len(edges_list)]):
                sdp_solution[i] = edges_weights[i]
        return  sdp_solution






    def brute_force_solution_4segments(self, G, list_len_segments, edges_list):

        cost_function_values = []
        edges_values_for_min_cost_function_values = []
        edges_positions_for_min_cost_function = []
        for i in range(0, list_len_segments[0]):
            for j in range(list_len_segments[0], list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):
                for k in range(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]), list_len_segments[0] + (
                        list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2])):
                    for m in range(list_len_segments[0] + (
                            list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]),
                                   list_len_segments[0] + (
                                           list_len_segments[0] * list_len_segments[1] +
                                           list_len_segments[1] * list_len_segments[2] + list_len_segments[2] * list_len_segments[3])):
                        for n in range(list_len_segments[0] + (
                                list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2] +
                                list_len_segments[2] * list_len_segments[3]), len(edges_list)):
                            v = [0, 0, 0, 0, 0]
                            edges_position = [0, 0, 0, 0, 0]
                            if edges_list[i][1] == edges_list[j][0] and edges_list[j][1] == edges_list[k][0] and \
                                    edges_list[k][
                                        1] == edges_list[m][0]  and  edges_list[m][1] == edges_list[n][0]:
                                v = [G[edges_list[i][0]][edges_list[i][1]]['weight'],
                                     G[edges_list[j][0]][edges_list[j][1]]['weight'],
                                     G[edges_list[k][0]][edges_list[k][1]]['weight'],
                                     G[edges_list[m][0]][edges_list[m][1]]['weight'],
                                     G[edges_list[n][0]][edges_list[n][1]]['weight']]
                                edges_position = [edges_list[i], edges_list[j], edges_list[k], edges_list[m], edges_list[n]]
                            cost_function_values = np.append(cost_function_values, Solver.cost_function(v, self.L))
                            edges_positions_for_min_cost_function.append([edges_position])
                            edges_values_for_min_cost_function_values.append([v])

        min = 1000
        for i in range(len(cost_function_values)):
            if cost_function_values[i] != 0 and cost_function_values[i] < min:
                min = cost_function_values[i]

        brute_force_solution = edges_values_for_min_cost_function_values[np.where(cost_function_values == min)[0][0]]
        positions = edges_positions_for_min_cost_function[np.where(cost_function_values == min)[0][0]]
        return brute_force_solution,positions



    def brute_force_solution_3segments(self, G, list_len_segments, edges_list):

        cost_function_values = []
        edges_values_for_min_cost_function_values = []
        edges_positions_for_min_cost_function = []

        for i in range(0, list_len_segments[0]):
            for j in range(list_len_segments[0], list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):
                for k in range(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]),
                               list_len_segments[0] + (
                                       list_len_segments[0] * list_len_segments[1] + list_len_segments[1] *
                                       list_len_segments[2])):
                    for m in range(list_len_segments[0] + (
                            list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]),
                                   len(edges_list)):

                        v = [0, 0, 0, 0]
                        edges_position = [0, 0, 0, 0, 0]
                        if edges_list[i][1] == edges_list[j][0] and edges_list[j][1] == edges_list[k][0] and edges_list[k][
                            1] == edges_list[m][0]:
                            v = [G[edges_list[i][0]][edges_list[i][1]]['weight'],
                                 G[edges_list[j][0]][edges_list[j][1]]['weight'],
                                 G[edges_list[k][0]][edges_list[k][1]]['weight'],
                                 G[edges_list[m][0]][edges_list[m][1]]['weight']]
                            edges_position = [edges_list[i], edges_list[j], edges_list[k], edges_list[m]]
                        cost_function_values = np.append(cost_function_values, Solver.cost_function(v, self.L))
                        edges_positions_for_min_cost_function.append([edges_position])
                        edges_values_for_min_cost_function_values.append([v])

        min = 1000
        for i in range(len(cost_function_values)):
            if cost_function_values[i] != 0 and cost_function_values[i] < min:
                min = cost_function_values[i]

        brute_force_solution = edges_values_for_min_cost_function_values[np.where(cost_function_values == min)[0][0]]
        positions = edges_positions_for_min_cost_function[np.where(cost_function_values == min)[0][0]]
        return brute_force_solution, positions

    '''
    def shortest_path_4segments(self, G, list_len_segments, edges_list):

        shortest_path_values = []
        edges_values_for_min_shortest_path_values = []

        for i in range(0, list_len_segments[0]):
            for j in range(list_len_segments[0], list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):
                for k in range(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]),
                               list_len_segments[0] + (
                                       list_len_segments[0] * list_len_segments[1] + list_len_segments[1] *
                                       list_len_segments[2])):
                    for m in range(list_len_segments[0] + (
                            list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]),
                                   list_len_segments[0] + (
                                           list_len_segments[0] * list_len_segments[1] +
                                           list_len_segments[1] * list_len_segments[2] + list_len_segments[2] *
                                           list_len_segments[3])):
                        for n in range(list_len_segments[0] + (
                                list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[
                            2] +
                                list_len_segments[2] * list_len_segments[3]), len(edges_list)):
                            v = [0, 0, 0, 0, 0]
                            if edges_list[i][1] == edges_list[j][0] and edges_list[j][1] == edges_list[k][0] and \
                                    edges_list[k][
                                        1] == edges_list[m][0]:
                                v = [G[edges_list[i][0]][edges_list[i][1]]['weight'],
                                     G[edges_list[j][0]][edges_list[j][1]]['weight'],
                                     G[edges_list[k][0]][edges_list[k][1]]['weight'],
                                     G[edges_list[m][0]][edges_list[m][1]]['weight'],
                                     G[edges_list[n][0]][edges_list[n][1]]['weight']]
                            shortest_path_values = np.append(shortest_path_values, Solver.shotest_path(v))

                            edges_values_for_min_shortest_path_values.append([v])

        min = 1000
        for i in range(len(shortest_path_values)):
            if shortest_path_values[i] != 0 and shortest_path_values[i] < min:
                min = shortest_path_values[i]

        shortest_path_solution = edges_values_for_min_shorthest_path_values[np.where(shortest_path_values == min)[0][0]]

        return shortest_path_solution


    def shortest_path_3segments(self, G, list_len_segments, edges_list):

        shortest_path_values = []
        edges_values_for_min_shortest_path_values = []

        for i in range(0, list_len_segments[0]):
            for j in range(list_len_segments[0], list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):
                for k in range(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]),
                               list_len_segments[0] + (
                                       list_len_segments[0] * list_len_segments[1] + list_len_segments[1] *
                                       list_len_segments[2])):
                    for m in range(list_len_segments[0] + (
                            list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]),
                                   len(edges_list)):

                        v = [0, 0, 0, 0]
                        if edges_list[i][1] == edges_list[j][0] and edges_list[j][1] == edges_list[k][0] and edges_list[k][
                            1] == edges_list[m][0]:
                            v = [G[edges_list[i][0]][edges_list[i][1]]['weight'],
                                 G[edges_list[j][0]][edges_list[j][1]]['weight'],
                                 G[edges_list[k][0]][edges_list[k][1]]['weight'],
                                 G[edges_list[m][0]][edges_list[m][1]]['weight']]
                        shortest_path_values = np.append(shortest_path_values, Solver.shortest_path(v, self.L))

                        edges_values_for_min_cost_function_values.append([v])

       min = 1000
        for i in range(len(shortest_path_values)):
            if shortest_path_values[i] != 0 and shortest_path_values[i] < min:
                min = shortest_path_values[i]

       shortest_path_solution = edges_values_for_min_shortest_path_values[np.where(shortest_path_values == min)[0][0]]

        return shortest_path_solution

'''






    @staticmethod
    def cost_function(v, L):
        return np.sum(v) + L * (np.sum([((v[i]-v[i-1])**2)/2 for i in range(1, len(v))]))

    '''
    @staticmethod
    def shortest_path(v):
        return np.sum(v)
'''

    def get_solution_rounding_2_4segments(self, G, z, list_len_segments):
        edges_list = list(G.edges)

        elements1 = np.sort(z[0:list_len_segments[0]])
        sdp_sol = np.zeros(len(z))
        for i in range(0, list_len_segments[0]):
            if z[i] == elements1[len(elements1) - 1]:
                sdp_sol[i] = z[i]
            if z[i] == elements1[len(elements1) - 2]:
                sdp_sol[i] = z[i]

        elements2 = np.sort(z[list_len_segments[0]:list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])])
        for i in range(list_len_segments[0], list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):
            if z[i] == elements2[len(elements2) - 1]:
                sdp_sol[i] = z[i]
            if z[i] == elements2[len(elements2) - 2]:
                sdp_sol[i] = z[i]

        elements3 = np.sort(z[list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]):list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2])])
        for i in range(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]), list_len_segments[0] + (
                list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2])):
            if z[i] == elements3[len(elements3) - 1]:
                sdp_sol[i] = z[i]
            if z[i] == elements3[len(elements3) - 2]:
                sdp_sol[i] = z[i]

        elements4 = np.sort(z[list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]):list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2] + list_len_segments[2] * list_len_segments[3])])
        for i in range(
                list_len_segments[0] + (list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]),
                list_len_segments[0] + (
                        list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2] + list_len_segments[2] * list_len_segments[3])):
            if z[i] == elements4[len(elements4) - 1]:
                sdp_sol[i] = z[i]
            if z[i] == elements4[len(elements4) - 2]:
                sdp_sol[i] = z[i]

        elements5 = np.sort(z[list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2] + list_len_segments[2] * list_len_segments[3]):len(
            edges_list)])
        for i in range(list_len_segments[0] + (
                list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2] + list_len_segments[2] * list_len_segments[3]),
                       len(edges_list)):
            if z[i] == elements5[len(elements5) - 1]:
                sdp_sol[i] = z[i]
            if z[i] == elements5[len(elements5) - 2]:
                sdp_sol[i] = z[i]

        edges = [edges_list[e] for e in np.where(sdp_sol != 0)[0]]

        tree = {}
        fill_tree(tree=tree, n=0, previous=(-1, 0), number_of_segments=self.number_of_segments, edges=edges)
        tree = {"root": tree}

        # we sum 2 -> one for root and one for virtual segment
        possible_paths = [p[1:] for p in paths(tree) if len(p) == self.number_of_segments + 2]

        if len(possible_paths) == 0:
            return 0, None

        cost_fn_values = [self.solution_cost_function_value(G, possible_paths[i]) for i in range(len(possible_paths))]
        best_solution_index = np.argmin(cost_fn_values)
        best_solution_value = np.min(cost_fn_values)

        return best_solution_value, possible_paths[best_solution_index]


    def get_solution_rounding_2_3segments(self, G, z, list_len_segments):
        edges_list = list(G.edges)

        elements1 = np.sort(z[0:list_len_segments[0]])
        sdp_sol = np.zeros(len(z))
        for i in range(0, list_len_segments[0]):
            if z[i] == elements1[len(elements1) - 1]:
                sdp_sol[i] = z[i]
            if z[i] == elements1[len(elements1) - 2]:
                sdp_sol[i] = z[i]

        elements2 = np.sort(z[list_len_segments[0]:list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])])
        for i in range(list_len_segments[0], list_len_segments[0] + (list_len_segments[0] * list_len_segments[1])):
            if z[i] == elements2[len(elements2) - 1]:
                sdp_sol[i] = z[i]
            if z[i] == elements2[len(elements2) - 2]:
                sdp_sol[i] = z[i]

        elements3 = np.sort(z[list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]):list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2])])
        for i in range(list_len_segments[0] + (list_len_segments[0] * list_len_segments[1]), list_len_segments[0] + (
                list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2])):
            if z[i] == elements3[len(elements3) - 1]:
                sdp_sol[i] = z[i]
            if z[i] == elements3[len(elements3) - 2]:
                sdp_sol[i] = z[i]

        elements4 = np.sort(z[list_len_segments[0] + (
                    list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]):len(edges_list)])
        for i in range(
                list_len_segments[0] + (list_len_segments[0] * list_len_segments[1] + list_len_segments[1] * list_len_segments[2]),
                len(edges_list)):
            if z[i] == elements4[len(elements4) - 1]:
                sdp_sol[i] = z[i]
            if z[i] == elements4[len(elements4) - 2]:
                sdp_sol[i] = z[i]


        edges = [edges_list[e] for e in np.where(sdp_sol != 0)[0]]

        tree = {}
        fill_tree(tree=tree, n=0, previous=(-1, 0), number_of_segments=self.number_of_segments, edges=edges)
        tree = {"root": tree}

        # we sum 2 -> one for root and one for virtual segment
        possible_paths = [p[1:] for p in paths(tree) if len(p) == self.number_of_segments + 2]

        if len(possible_paths) == 0:
            return 0, None

        cost_fn_values = [self.solution_cost_function_value(G, possible_paths[i]) for i in range(len(possible_paths))]
        best_solution_index = np.argmin(cost_fn_values)
        best_solution_value = np.min(cost_fn_values)

        return best_solution_value, possible_paths[best_solution_index]





if __name__ == '__main__':

    len_first_segment = 3
    len_second_segment = 2
    len_third_segment = 2
    len_fourth_segment = 3

    list_len_segments = [len_first_segment,
                         len_second_segment, len_third_segment, len_fourth_segment]
    if len(list_len_segments) == 4:
        G = Solver.make_graph_4segments(list_len_segments=list_len_segments)
    elif len(list_len_segments) == 3:
        G = Solver.make_graph_3segments(list_len_segments=list_len_segments)

    solver = Solver(L=1.0)
    sdp_solution, brute_force_solution, brute_force_positions, W, rounding2_solution = solver.solve(G=G, list_len_segments=list_len_segments)

    edges_list = list(G.edges)
    path_edges = [edges_list[e] for e in np.where(~np.isnan(sdp_solution))[0]]
    path_nodes = sorted(list(set([item for sublist in path_edges for item in sublist])))

    print("SDP Solution:")
    print(path_edges)
    print(solver.solution_cost_function_value(G,path_edges))

    brute_force_solution = np.array(brute_force_solution[0])
    print("BruteForce Solution:")
    print(brute_force_solution, brute_force_positions )
    print(solver.solution_cost_function_value(G,brute_force_positions[0]))

    print("Rounding 2 Solution:")
    print(rounding2_solution)
    if rounding2_solution is not None:
        print(solver.solution_cost_function_value(G, rounding2_solution))
    else:
        print("No solution")

    solver = Solver(L=0.0)
    sdp_solution, brute_force_solution, brute_force_positions, W, rounding2_solution = solver.solve(G=G,
                                                                                                    list_len_segments=list_len_segments)

    brute_force_solution = np.array(brute_force_solution[0])
    print("Shortest Path Solution:")
    print(brute_force_solution, brute_force_positions )
    print(solver.solution_cost_function_value(G,brute_force_positions[0]))