
from bayesianNetwork import BeliefNetwork

import numpy as np
import time
import os
from episode import Episode


"""
This class models a physical Softbank robot (NAO or Pepper).
Programmed on NAOqi version 2.5.5.
"""

DIR_NAME=os.path.dirname(__file__)

class Robot:
    def __init__(self, user):
       
        
        self.informants = user
        name="Informer" + user + ".csv"
        path_name=os.path.join(DIR_NAME,"datasets", name)
        
        self.beliefs = []
        self.time = None
        self.load_time()
        print(path_name, os.path.isfile(path_name), self.beliefs==[])
        #if self.beliefs==[]:
        if os.path.isfile(path_name):
                print("Hiiiiiiii")
                self.beliefs.append(BeliefNetwork("Informer" + str(user), path_name))
        print(self.beliefs, "rrrrr")
        if self.beliefs==[]:

            demo_result=[]
            demo_sample_episode = Episode([1,1,1,1], self.get_and_inc_time())
            demo_result.append(demo_sample_episode)
            demo_sample_episode = Episode([0,0,0,0], self.get_and_inc_time())
            demo_result.append(demo_sample_episode)
            demo_sample_episode = Episode([1,1,1,0], self.get_and_inc_time())
            demo_result.append(demo_sample_episode)
            demo_sample_episode = Episode([0,0,0,1], self.get_and_inc_time())
            demo_result.append(demo_sample_episode)
            self.beliefs.append(BeliefNetwork("Informer" + str(user), demo_result))
        print(self.beliefs)
        
        self.load_time()
        





    # Manages the unknown informant detection
    def manage_unknown_informant(self):
        # Updates the model with the acquired frames and the right label
        
        # Creates an episodic belief network
        name = "Informer" + str(self.informants) + "_episodic"
        episodic_network = BeliefNetwork.create_episodic(self.beliefs, self.get_and_inc_time(), name=name)
        self.beliefs.append(episodic_network)
        # Updates the total of known informants



    
    # Load time value from file
    def load_time(self):
        if os.path.isfile("current_time.csv"):
            with open("current_time.csv", 'r') as f:
                self.time = int(f.readline())
        else:
            self.time = 0

    # Increases and saves the current time value
    def get_and_inc_time(self):
        previous_time = self.time
        self.time += 1
        with open("current_time.csv", 'w') as f:
            f.write(str(self.time))
        return previous_time

    # Saves the beliefs
    def save_beliefs(self):
        if not os.path.exists(os.path.join(DIR_NAME,"datasets")):
            os.makedirs(os.path.join(DIR_NAME,"datasets"))
        for belief in self.beliefs:
            belief.save()

    # Loads the beliefs
    def load_beliefs(self, path=os.path.join(DIR_NAME,"datasets")):
        # Resets previous beliefs
        self.beliefs = []
        i = 0
        while os.path.isfile(path + "Informer" + str(i) + ".csv"):
            self.beliefs.append(BeliefNetwork("Informer" + str(i), path + "Informer" + str(i) + ".csv"))
            i += 1

    # Reset time
    def reset_time(self):
        if os.path.isfile("current_time.csv"):
            with open("current_time.csv", 'w') as f:
                f.write("0")
        self.time = 0
