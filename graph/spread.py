# TODO: I need to add influecne maximization to this class

import numpy as np
import networkx as nx

from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt


class SIR:
    def __init__(self, graph, beta=0.1, mu=1.0, gamma=0.0, seed=None):
        self.graph = graph
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
        np.random.seed(seed)
        self.itr = 0
        self.spread = 0.0
        self.status = {n: 0 for n in self.graph.nodes()}
        self.I_set = set()
        self.S_set = set(self.graph.nodes())
        self.R_set = set()
        self.B_set = set()  # Blocked i.e. immunied
        self.parameter = {
            'beta': 'Infection rate',
            'mu': 'Recovery rate',
            'gamma': 'Immunization rate',
            0: 'Susceptible',
            1: 'Infected',
            2: 'Recovered',
            3: 'Immunized',
        }

    def reset(self):
        self.itr = 0
        self.spread = 0.0
        self.status = {n: 0 for n in self.graph.nodes()}
        self.I_set = set()
        self.S_set = set(self.graph.nodes())
        self.R_set = set()
        self.B_set = set()

    def characteristic_time(self):
        # Time that 1/e fraction (about 36%) of all susceptible individuals
        # Become Infected based on average degree of the graph
        average_degree = sum(dict(self.graph.degree()).values()
                             ) / self.graph.number_of_nodes()
        tau = 1 / self.beta * average_degree
        return tau

    def epidemic_threshold(self):
        # Epidemic threshold i.e. beta_c
        average_degree = sum(dict(self.graph.degree()).values()
                             ) / self.graph.number_of_nodes()
        average_degree_2 = np.mean(
            [x**2 for x in dict(self.graph.degree()).values()]
        )
        beta_c = average_degree / (average_degree_2 - average_degree)
        return beta_c

    def set_beta(self, ratio=1.5, output=False):
        beta_c = self.epidemic_threshold()
        self.beta = ratio * beta_c
        if output: print(f'beta = {ratio} * beta_c = {self.beta}')

    def infect(self):
        return True if np.random.uniform(0.0, 1.0) < self.beta else False

    def recover(self):
        return True if np.random.uniform(0.0, 1.0) < self.mu else False

    def update_S(self):
        # (1) Susceptible excludes recovered
        # Recovered nodes won't get infected again = SIR
        self.S_set = set(self.graph.nodes()).difference(self.I_set).difference(
            self.R_set
        ).difference(self.B_set)
        # (2) Susceptible includes recovered
        # Recovered nodes can get infected again = SIS
        # self.S_set = set(self.graph.nodes()).difference(self.I_set).difference(self.B_set)

    def update_I(self, I_new=None):
        if I_new is not None:
            # Add newly infected to infected set
            self.I_set.update(I_new)

    def update_R(self, R_new=None):
        if R_new is not None:
            # Remove recovered from infected set
            self.I_set = self.I_set.difference(R_new)
            # Add recovered nodes to recovered group
            self.R_set.update(R_new)
            # SIS = Add recovered nodes back to susceptible group

    def update_spread(self):
        self.spread = len(self.R_set) / self.graph.number_of_nodes()

    def start(self, infected=None, infected_num=1, output=False):
        # At the beginning of simulation
        # Infect first node(s) to infect, from the input list or randomly
        if self.itr == 0:
            if infected is None:
                if infected_num < 1:
                    # Infected ratio is set
                    infected_num = int(
                        infected_num * self.graph.number_of_nodes()
                    )
                infected = list(
                    np.random.choice(
                        self.graph.nodes(), size=infected_num, replace=False
                    )
                )
            self.I_set.update(infected)
            self.S_set.difference(infected)
            for node in infected:
                self.status[node] = 1
            if output:
                print('Iteration = 0')
                # print('Number of Infected =', len(self.I_set))
                print('Patient zero =', self.I_set)
                print('---')

    def run(
        self,
        steps=1,
        infected=None,
        recovered=None,
        immunized=None,
        output=False,
    ):
        # Manually add new infected nodes
        if infected is not None:
            self.I_set.update(infected)

        # Manually recover some of infected nodes
        if recovered is not None:
            self.R_set.update(recovered)

        # Manually immunie or heal some of nodes
        if immunized is not None:
            self.B_set.update(immunized)
        else:
            # Randomly immune node(s) based on gamma
            immunized = list(
                np.random.choice(
                    self.graph.nodes(),
                    size=int(self.gamma * self.graph.number_of_nodes()),
                    replace=False
                )
            )

        # Step counter of this run
        t = 0
        while len(self.I_set) > 0:
            # Current infected nodes
            I = self.I_set.copy()
            # Nodes that may recover
            R = set()
            StoI = set()
            ItoR = set()
            # Start infecttion
            for i in I:
                # Look at neighbors of infected node
                for n in set(self.graph.neighbors(i)).intersection(self.S_set):
                    if self.infect():
                        # Add to newly infected set
                        StoI.add(n)
                # Node may recover later based on mu parameter
                R.add(i)
            # Start recovery
            for i in R:
                if self.recover():
                    ItoR.add(i)
            # At this point, 1 time step is completed
            # Update variables
            self.update_I(StoI)
            self.update_R(ItoR)
            # It is important to update S after I and R
            self.update_S()
            self.update_spread()
            t += 1
            self.itr += 1
            # Stop IF ... reach the number of steps
            # Meet desired rocovered ratio or all node recoveres or no more infections
            if t % steps == 0 or len(self.I_set) == 0:
                # Output result
                if output:
                    print(f'Iteration = {self.itr}')
                    print(f'Number of Susceptible = {len(self.S_set)}')
                    print(f'Number of Infected = {len(self.I_set)}')
                    print(f'Number of Recovered = {len(self.R_set)}')
                    # print(f'Spread = {np.round(self.spread * 100,2)} %')
                    print(f'Spread = {self.spread * 100} %')
                    print('---')
                    # Report back the measures
                yield (
                    {
                        't': self.itr,
                        's': self.spread,
                        'S': self.S_set,
                        'I': self.I_set,
                        'R': self.R_set,
                        'StoI': StoI,
                        'ItoR': ItoR,
                    }
                )

    def simulate(self, p_0=None, steps=1, repeat=10):
        # Patient zero i.e. p_0 is initial infected node at the start
        results = []
        for i in range(0, repeat):
            if p_0 is not None:
                self.start(infected=p_0, output=True)
            else:
                self.start(output=True)  # Radnom patient zero
            simulation_step = [
                r['s'] for r in self.run(steps=steps, output=True)
            ]
            results.append(simulation_step[len(simulation_step) - 1])
            self.reset()
        return results

    def simulate_r(self, steps=1, repeat=10, plot=True):
        # Set every node as patient zero at every iteration
        # Average the spread score over several simulation (e.g. 10)
        scores = {}
        for node in self.graph.nodes():
            results = []
            for i in range(0, repeat):
                self.start(infected=[node])
                simulation_step = [r['s'] for r in self.run(steps=steps)]
                # Spread result from last time step
                # Where there is no infected left
                results.append(simulation_step[len(simulation_step) - 1])
                self.reset()
            scores[node] = np.mean(results)
        # Sort the scores
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
        if plot:
            plt.figure(figsize=(12, 6), dpi=400)
            x, y = zip(*sorted_scores)
        return scores

    def simulate_sir(self, steps=1, repeat=10):
        # Set every node as patient zero at every iteration
        # Average the spread score over several simulation (e.g. 10)
        # Set plot settings.
        plt.figure(figsize=(12, 6), dpi=400)
        N = self.graph.number_of_nodes()
        resutls = {}
        for node in self.graph.nodes():
            # results = []
            s_t = defaultdict(list)
            i_t = defaultdict(list)
            r_t = defaultdict(list)
            for i in range(0, repeat):
                self.start(infected=[node])
                for r in self.run(steps=steps):
                    s_t[r['t']].append(len(r['S']))
                    i_t[r['t']].append(len(r['I']))
                    r_t[r['t']].append(len(r['R']))
                self.reset()
            # Now time to average the results for each time step
            s_t = {k: np.mean(v) / N for k, v in s_t.items()}
            i_t = {k: np.mean(v) / N for k, v in i_t.items()}
            r_t = {k: np.mean(v) / N for k, v in r_t.items()}
            plt.plot(
                s_t.keys(),
                s_t.values(),
                color='blue',
                linewidth=0.5,
                alpha=0.5
            )
            plt.plot(
                i_t.keys(),
                i_t.values(),
                color='red',
                linewidth=0.5,
                alpha=0.5
            )
            plt.plot(
                r_t.keys(),
                r_t.values(),
                color='green',
                linewidth=0.5,
                alpha=0.5
            )
            resutls[node] = {'st': s_t, 'it': i_t, 'rt': r_t}
        plt.title("SIR Simulation")
        # plt.legend(['S', 'I', 'R'])
        plt.xlabel("Time")
        plt.ylabel("Ratio of SIR nodes")
        plt.show()
        return resutls
