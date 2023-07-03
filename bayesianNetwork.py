import copy
import os
from math import log
from random import randint, shuffle, choice

#from pybbn import *
import pandas as pd


from datasetParser import DatasetParser
from episode import Episode
# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs

"""
This class models the Developmental Bayesian Model of Trust in Artificial Cognitive Systems (Patacchiola, 2016).
Methods include the generation of the Episodic Memory by importance filtering and systematic resampling.
"""

DIR_NAME=os.path.dirname(__file__)

class BeliefNetwork:
    def __init__(self, name, dataset):
        self.name = name
        print("HHI")
        self.dataset = DatasetParser(dataset)
        print("Hello")
        self.parameters = self.dataset.estimate_bn_parameters()
        self.bn = None
        self.join_tree = None
        
        # truth_a :
        # lie_a : sticker in A, informar said B
        # ecc...
        # Goal Reached: A
        # Goal not Reached B
        self.pdf = {
            'truth_a': 1.0,  # Robot thinks  reached goal, Human said goal reached
            'truth_b': 1.0,  # Robot thinks  did not reach goal, Human said goal not reached
            'lie_a': 1.0,    # Robot thinks  reached goal, Human said goal not reached
            'lie_b': 1.0     # Robot thinks did not reach goal, Human said goal reached
        }
        self.entropy = None
        # Post-initialization processes
        self.build()
        self.calculate_pdf()
        self.entropy = self.get_entropy()


    # Xi
    def f_informant_belief(self, informant_belief):
        # print(self.parameters)
        # sys.exit()
        if informant_belief == 'A':
            return self.parameters["Xi"][1]
        return self.parameters["Xi"][0]

    # Xr
    def f_robot_belief(self, robot_belief):
        if robot_belief == 'A':
            return self.parameters["Xr"][1]
        return self.parameters["Xr"][0]

    # Yi
    def f_informant_action(self, informant_belief, informant_action):
        if informant_belief == 'A' and informant_action == 'A':
            return self.parameters["Yi"][0][0]
        if informant_belief == 'A' and informant_action == 'B':
            return self.parameters["Yi"][0][1]
        if informant_belief == 'B' and informant_action == 'A':
            return self.parameters["Yi"][1][0]
        if informant_belief == 'B' and informant_action == 'B':
            return self.parameters["Yi"][1][1]

    # Yr
    def f_robot_action(self, informant_action, robot_belief, robot_action):
        table = dict()
        table['aaa'] = self.parameters["Yr"][0][0]
        table['aab'] = self.parameters["Yr"][0][1]
        table['aba'] = self.parameters["Yr"][1][0]
        table['abb'] = self.parameters["Yr"][1][1]
        table['baa'] = self.parameters["Yr"][2][0]
        table['bab'] = self.parameters["Yr"][2][1]
        table['bba'] = self.parameters["Yr"][3][0]
        table['bbb'] = self.parameters["Yr"][3][1]
        return table[make_key(informant_action, robot_belief, robot_action)]

    # Constructs the bayesian belief network
    def build(self):
        # self.bn = build_bbn(
        #     self.f_informant_belief,
        #     self.f_robot_belief,
        #     self.f_informant_action,
        #     self.f_robot_action,
        #     domains=dict(
        #         informant_belief=['A', 'B'],
        #         robot_belief=['A', 'B'],
        #         informant_action=['A', 'B'],
        #         robot_action=['A', 'B']),
        #     name=self.name
        # )
        # print(self.bn)
        #print(self.parameters )
        def probs(data, child, parent1=None, parent2=None):
            if parent1==None:
                # Calculate probabilities
                prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
            elif parent1!=None:
                    # Check if child node has 1 parent or 2 parents
                    if parent2==None:
                        # Caclucate probabilities
                        prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
                    else:    
                        # Caclucate probabilities
                        prob=pd.crosstab([data[parent1],data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else: print("Error in Probability Frequency Calculations")
            return prob  
       # print(self.parameters["Xi"], self.parameters["Yr"], self.parameters["Yi"], self.parameters["Xr"])
        y_r_p =[]
        for i in self.parameters["Yr"]:
            if type(i) is list:
                y_r_p.extend(i)
            else:
                y_r_p.append(i)
        y_i_p =[]
        for i in self.parameters["Yi"]:
            if type(i) is list:
                y_i_p.extend(i)
            else:
                y_i_p.append(i)
        #print(self.parameters["Xi"], self.parameters["Yi"])
        XI = BbnNode(Variable(0, 'informant_belief', ['A', 'B']), self.parameters["Xi"])
        YI = BbnNode(Variable(1, 'informant_action', ['A', 'B']), y_i_p)
        XR = BbnNode(Variable(2, 'robot_belief', ['A', 'B']), self.parameters["Xr"])
        YR = BbnNode(Variable(3, 'robot_action', ['A', 'B']), y_r_p)

        # Create Network
        self.bn = Bbn() \
            .add_node(XI) \
            .add_node(YI) \
            .add_node(XR) \
            .add_node(YR) \
            .add_edge(Edge(XI, YI, EdgeType.DIRECTED)) \
            .add_edge(Edge(XR, YR, EdgeType.DIRECTED)) \
            .add_edge(Edge(YI, YR, EdgeType.DIRECTED))

        # Convert the BBN to a join tree
        #print(self.bn)
        self.join_tree = InferenceController.apply(self.bn)
        # Set node positions
        pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}

        # Set options for graph looks
        options = {
            "font_size": 16,
            "node_size": 20000,
            "node_color": "white",
            "edgecolors": "black",
            "edge_color": "red",
            "linewidths": 5,
            "width": 5,}
            
        # Generate graph
        # n, d = self.bn.to_nx_graph()
        # nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

        # # Update margins and print the graph
        # ax = plt.gca()
        # ax.margins(0.10)
        # plt.axis("off")
        # plt.show()

    # Test query
    def test_query(self, prettyTable=False):
        if prettyTable:
            self.bn.q()
        return self.join_tree.query()

    # Decision Making
    # Sets informant_action as evidence and infers robot_action
    def decision_making(self, informant_action):
        if informant_action != 'A' and informant_action != 'B':
            return None
        else:

            # query the probability of node b being 'true'
           # b_prob = InferenceController.query(bbn, ['robot_action'], {})
            #print('Probability of b being true given evidence for a being true: {}'.format(b_prob))
            #outputs = self.join_tree.query(self.bn, ['robot_action'], {'informant_action':informant_action})
            def evidence(ev, nod, cat, val):
                ev = EvidenceBuilder() \
                .with_node(self.join_tree.get_bbn_node_by_name(nod)) \
                .with_evidence(cat, val) \
                .build()
                self.join_tree.set_observation(ev)
            evidence('ev1', 'informant_action', informant_action, 1.0)
            #print(outputs)
            outputs={'robot_action': {'A': 0, 'B':0 }}
            for node in self.join_tree.get_bbn_nodes():
                potential = self.join_tree.get_bbn_potential(node)
                print("Node:", node)
                print("Values:")
                print(potential)
                if node.to_dict()['variable']['name'] in  outputs:
                    outputs[node.to_dict()['variable']['name']][potential.entries[0].get_entry_keys()[0][1]] = potential.entries[0].get_kv()[1]#, help(node))

                    outputs[node.to_dict()['variable']['name']][potential.entries[1].get_entry_keys()[0][1]] = potential.entries[1].get_kv()[1]#, help(node))
            #print(outputs)
            if outputs['robot_action']['A'] > outputs['robot_action']['B']:
                return 'A'
            else:
                # If probability is 0.5, then picks a random choice
                if outputs['robot_action']['A'] == outputs['robot_action']['B']:
                    return choice(['A', 'B'])
                else:
                    return 'B'

    # Belief Estimation
    # Sets robot_belief and robot_action as evidence and infers informant_belief
    def belief_estimation(self, robot_knowledge):
        if robot_knowledge != 'A' and robot_knowledge != 'B':
            return None
        else:
            
            # outputs = self.join_tree.query(self.bn, ['informant_belief', 'informant_action'],\
            #  {'robot_belief': robot_knowledge, 'robot_action':robot_knowledge})
            
            def evidence(ev, nod, cat, val):
                ev = EvidenceBuilder() \
                .with_node(self.join_tree.get_bbn_node_by_name(nod)) \
                .with_evidence(cat, val) \
                .build()
                self.join_tree.set_observation(ev)
            evidence('ev1', 'robot_belief', robot_knowledge, 1.0)
            evidence('ev1', 'robot_action', robot_knowledge, 1.0)
            #print(outputs)
            outputs={'informant_belief': {'A': 0, 'B':0 }, 'informant_action': {'A': 0, 'B':0 }}
            for node in self.join_tree.get_bbn_nodes():
                potential = self.join_tree.get_bbn_potential(node)
                print("Node:", node)
                print("Values:")
                print(potential)
                if node.to_dict()['variable']['name'] in  outputs:
                    outputs[node.to_dict()['variable']['name']][potential.entries[0].get_entry_keys()[0][1]] = potential.entries[0].get_kv()[1]#, help(node))

                    outputs[node.to_dict()['variable']['name']][potential.entries[1].get_entry_keys()[0][1]] = potential.entries[1].get_kv()[1]#, help(node))
            #print(outputs)
            
           # outputs = self.join_tree.query(robot_belief=robot_knowledge, robot_action=robot_knowledge)
            # Duple: [informant_belief , informant_action]
            predicted_behaviour = []
            if outputs['informant_belief']['A'] > outputs['informant_belief']['B']:
                predicted_behaviour.append('A')
            else:
                # If probability is equals, pick at random
                if outputs['informant_belief']['A'] == outputs['informant_belief']['B']:
                    predicted_behaviour.append(choice(['A', 'B']))
                else:
                    predicted_behaviour.append('B')
            if outputs['informant_action']['A'] > outputs['informant_action']['B']:
                predicted_behaviour.append('A')
            else:
                if outputs['informant_action']['A'] == outputs['informant_action']['B']:
                    predicted_behaviour.append(choice(['A', 'B']))
                else:
                    predicted_behaviour.append('B')
            return predicted_behaviour

    # Updates in real-time the belief
    def update_belief(self, new_data):
        if isinstance(new_data, Episode):
            previous_dataset = self.get_episode_dataset()
            previous_dataset.append(new_data)   # "previous_dataset" is now updated with new data
            self.dataset = DatasetParser(previous_dataset)
            self.parameters = self.dataset.estimate_bn_parameters()
            self.build()
            self.calculate_pdf()
        else:
            print ("[ERROR] update_belief: new data is not an Episode instance.")
            quit(-1)

    # Prints the network parameters
    def print_parameters(self):
        print (self.name + "\n" + str(self.parameters))

    # Gets the raw input dataset of the BN
    def get_episode_dataset(self):
        return self.dataset.episode_dataset

    # Static Method
    # Creates a new BN used for episodic memory. Uses all the previous information collected
    @staticmethod
    def create_full_episodic_bn(bn_list, time):
        dataset = []
        for bn in bn_list:
            episode_list = bn.get_episode_dataset()
            for episode in episode_list:
                dataset.append(Episode(episode.raw_data, time))
        episodic_bn = BeliefNetwork("Episodic", dataset)
        return episodic_bn

    # Saves the BN's dataset for future reconstruction
    def save(self, path=os.path.join(DIR_NAME,"datasets")):
        if not os.path.isdir(path):
            os.makedirs(path)
        
        self.dataset.save(os.path.join(path, self.name + ".csv"))

    # Calculates the probability distribution
    def calculate_pdf(self):
        n = len(self.dataset.episode_dataset) + 4   # The 4 factor compensates the initialization at 1 insted of 0
        # Resets all the pdf values
        self.pdf = self.pdf.fromkeys(self.pdf, 1.0)
        for item in self.dataset.episode_dataset:
            item = item.raw_data
            if item == [1, 1, 1, 1]:
                self.pdf['truth_a'] += 1.0
            elif item == [0, 0, 0, 0]:
                self.pdf['truth_b'] += 1.0
            elif item == [0, 0, 0, 1]:
                self.pdf['lie_a'] += 1.0
            elif item == [1, 1, 1, 0]:
                self.pdf['lie_b'] += 1.0
            else:
                print ("[ERROR] invalid dataset item not counted: " + str(item))
        for key in self.pdf:
            self.pdf[key] /= n

    # Calculates the information entrophy
    def get_entropy(self):
        entropy = 0
        for key, Px in self.pdf.items():
            if Px != 0:
                entropy += Px * log(Px, 2)
        return -1 * entropy

    # Calculates the unlikelihood of an episode in terms of information theory
    def surprise(self, episode):
        return log(1 / self.pdf[episode.get_label()], 2)

    # Distance between the episode's surprise value and the network's entropy
    def entropy_difference(self, episode):
        normalization_factor = 2
        entropy = self.get_entropy()
        surprise = self.surprise(episode)
        return round(abs(surprise - entropy), 1) / normalization_factor

    # Importance sampling: calculates how many copies of each episod need to be generated
    def importance_sampling(self, episode, time):
        mitigation_factor = 2.0
        entropy_diff = self.entropy_difference(episode)
        time_fading = (time - episode.time + 1) / mitigation_factor
        consistency = entropy_diff / time_fading
        if 0.0 <= consistency <= 0.005:
            duplication_value = 0
        elif 0.005 < consistency <= 0.3:
            duplication_value = 1
        elif 0.3 < consistency <= 0.6:
            duplication_value = 2
        else:
            duplication_value = 3
        return [episode] * duplication_value

    # Systematic Resampling
    @staticmethod
    def systematic_resampling(samples, to_generate=10):
        output = []
        sample_size = len(samples)
        x = randint(0, sample_size)
        increment = randint(2, sample_size-1)     # Avoids increments with undesirable effects (0, 1, len)
        for i in range(to_generate):
            output.append(samples[x % sample_size])
            x += increment
        return output

    # Creates an episodic belief network based on previous beliefs
    @staticmethod
    def create_episodic(bn_list, time, generated_episodes=6, name="EpisodicMemory"):
        weighted_samples = []
        for bn in bn_list:
            episode_list = bn.get_episode_dataset()
            for episode in episode_list:
                samples = bn.importance_sampling(episode, time)
                if samples:
                    weighted_samples.append(samples)
        # Flattens out list of lists
        weighted_samples = [item for sublist in weighted_samples for item in sublist]
        # Checks that there are enough samples to produce a systematic resampling
        if len(weighted_samples) < 4:
            print ("create_episodic: not enough samples. Needed at least 4, found " + str(len(weighted_samples)))
            quit()
        # Shuffles the list to prevent the first items to be the most likely to be selected
        shuffle(weighted_samples)
        # Now peform Systematic Resampling
        dataset = BeliefNetwork.systematic_resampling(weighted_samples, to_generate=generated_episodes)
        # Copy the list without reference (avoids timing changes to the original episodes)
        unreferenced_dataset = copy.deepcopy(dataset)
        output_dataset = []
        for sample in unreferenced_dataset:
            # Change the time value of all the samples to the current one
            sample.time = time
            output_dataset.append(sample)
            # Generate the symmetric episode
            output_dataset.append(sample.generate_symmetric())
        return BeliefNetwork(name, output_dataset)

    # Returns the reliability of the network as a real value between -1 and +1. Negative values denote degrees of
    # unreliability and vice versa.
    def get_reliability(self, goal_status):
        # Evaluate the trustworthiness of the network (as for Belief Estimation)
        outputs={'informant_belief': {'A': 0, 'B':0 }, 'informant_action': {'A': 0, 'B':0 },
                 'robot_belief': {'A': 0, 'B':0 }, 'robot_action': {'A': 0, 'B':0 }}
        def evidence(ev, nod, cat, val):
                ev = EvidenceBuilder() \
                .with_node(self.join_tree.get_bbn_node_by_name(nod)) \
                .with_evidence(cat, val) \
                .build()
                self.join_tree.set_observation(ev)
        evidence('ev1', 'robot_belief', 'A', 1.0)
        evidence('ev2', 'robot_action', 'A', 1.0)
        for node in self.join_tree.get_bbn_nodes():
            potential = self.join_tree.get_bbn_potential(node)
            print("Node:", node)
            print("Values:")
            print(potential)
            if node.to_dict()['variable']['name'] in  outputs:
                outputs[node.to_dict()['variable']['name']][potential.entries[0].get_entry_keys()[0][1]] = potential.entries[0].get_kv()[1]#, help(node))

                outputs[node.to_dict()['variable']['name']][potential.entries[1].get_entry_keys()[0][1]] = potential.entries[1].get_kv()[1]#, help(node))
        
       # be_query = self.join_tree.query(self.bn, ['informant_action'], {'robot_belief':'A', 'robot_action':'A'})
        x = outputs['informant_action'][goal_status]
        # Scale it to [-1, +1]
        a = -1
        b = 1
        min = 0
        max = 0.99
        
        return ((b - a) * (x - min)) / (max - min) + a
