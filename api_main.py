import time

import random
from bayesianNetwork import BeliefNetwork
from episode import Episode
from robot import Robot
import pandas as pd
import os
DIR_NAME=os.path.dirname(__file__)
class APITrust2:


    def __init__(self, user, mature=True, simulation=False, withUpdate=True) -> None:
        
        
        print("Familiarization Phase " + user)
        
        self.user = 0

        self.informant_vocabulary = ["reached", "not reached"]
        self.mature = mature
        self.simulation = simulation
        self.withUpdate = withUpdate
        self.trust = 0
        self.p = 0
        self.robot=Robot(user)
        #if  BeliefNetwork("Informer" + str(self.user))get_episode_dataset
       
        
        self.csv_file_path =os.path.join(DIR_NAME, user + "apibaseyianless.csv")
        self.cols = ["robot_beliefA", "robot_actionA", "informant_actionA", "informant_beliefA", \
                     "robot_beliefB", "robot_actionB", "informant_actionB", "informant_beliefB", "Robot Opinion",
                     "Robot Goal Status", "Human Goal Status", "Trust", "Post P"]
        if not os.path.isfile(self.csv_file_path):
            self.data=pd.DataFrame([],columns=self.cols)
            self.data.to_csv(self.csv_file_path,index=False)
        else:
            self.data = pd.read_csv(self.csv_file_path)
        self.robot_opinion = ""
        #BeliefNetwork("Informer" + str(self.user), demo_result).create_full_episodic_bn()

    def start(self, human_goal_status, robot_goal_status):
        if human_goal_status:
            hint="A"
        else:
            hint="B"
        if robot_goal_status:
            goal_status = "A"
        else:
            goal_status = "B"
        print("Decision Making Phase")
        repeat = "yes"
        # for i in range(50):
        #     repeat = "yes"
        #     while repeat == "yes":
        count = 1
        #while count<random.randint(1, 10):
        self.decision_making(hint, goal_status, withUpdate=self.withUpdate)
         #   count=count+1
        # print("Do you want to repeat? Yes or no.")
        # repeat = random.choice(["Yes", "No"])
        # repeat = 'No'
        # print("Ok then, let's continue with the experiment.")
        # time.sleep(2)
        # print("Belief Estimation Phase")
        # repeat = "yes"
        count = 1
        #while count<random.randint(1, 10):
        self.belief_estimation(goal_status)
         #   count=count+1
        # print("Do you want to repeat? Yes or no.")
        # repeat = "yes"
        # repeat = 'No'
        # #count = count +1
        self.get_node_values(hint, goal_status)
            
            
        print("The experiment has ended. Thank you for your participation.")
        self.robot.save_beliefs()
        

    def get_node_values(self, hint, goal_status):
        informer = self.user
        outputs={'informant_belief': {'A': 0, 'B':0 }, 'informant_action': {'A': 0, 'B':0 },
                 'robot_belief': {'A': 0, 'B':0 }, 'robot_action': {'A': 0, 'B':0 }}
        for node in self.robot.beliefs[informer].join_tree.get_bbn_nodes():
            potential = self.robot.beliefs[informer].join_tree.get_bbn_potential(node)
            print("Node:", node)
            print("Values:")
            print(potential)
            if node.to_dict()['variable']['name'] in  outputs:
                outputs[node.to_dict()['variable']['name']][potential.entries[0].get_entry_keys()[0][1]] = potential.entries[0].get_kv()[1]#, help(node))

                outputs[node.to_dict()['variable']['name']][potential.entries[1].get_entry_keys()[0][1]] = potential.entries[1].get_kv()[1]#, help(node))
        
        min_x=0.25
        max_x=0.75
        a=-1
        b=1
        p=outputs['informant_action']['A']
             
        #trust_p = ((b-a)*(p-min_x)/(max_x-min_x)) + a
        trust_p = self.robot.beliefs[informer].get_reliability()
        print(trust_p, outputs, "fffff", p)
        self.trust = trust_p
        self.p=p
        self.data = pd.concat([self.data, pd.DataFrame([[outputs['robot_belief']["A"], outputs['robot_action']["A"],
        outputs['informant_belief']["A"],outputs['informant_action']["A"],\
        outputs['robot_belief']["B"],outputs['robot_action']["B"], \
        outputs['informant_belief']["B"],outputs['informant_action']["B"], self.robot_opinion,
        hint, goal_status, self.trust, self.p]], columns=self.cols)])
        self.data.to_csv(self.csv_file_path, index=False)  

    # Decision Making Phase
    def decision_making(self, hint, goal_status, withUpdate=True):
        # The robot first recognizes the informer
        informer = self.user
        
        print("Can you suggest me the Goal status? A or B?")
        #hint = random.choice(["A", "B"])
        # A = Goal Reached
        # B = Goal not Reached
        # Decision making based on the belief network for that particular informant
        choice = self.robot.beliefs[informer].decision_making(hint)
        # if self.simulation:
        #     print ("Robot decides to look at position: " + str(choice))
        # else:
        #     print("I'm thinking at where to look based on your suggestion...")
        #     self.robot.animation_service.runTag("think")
        #     self.robot.set_led_color("white")
        #goal_status = random.choice(["A", "B"]) 
        found = False
        if goal_status == "A":
            found=True# goal status as per robot
        if self.mature:
            # Mature ToM
            if hint == choice and found:
                print("I trusted you and your suggestion was correct. Thank you!")
                self.robot_opinion = "I trusted you and your suggestion was correct. Thank you!"
                print("friendly")
            elif hint == choice and not found:
                print("I trusted you, but you tricked me.")
                self.robot_opinion = "I trusted you, but you tricked me."
                print("frustrated")
            elif hint != choice and found:
                print("I was right not to trust you.")
                self.robot_opinion = "I was right not to trust you."
                print("indicate")
            elif hint != choice and not found:
                print("I didn't trust you, but I was wrong. Sorry.")
                self.robot_opinion = "I didn't trust you, but I was wrong. Sorry."
                print("ashamed")
        else:
            # Immature ToM
            if found:
                print("Oh, here it is!")
            else:
                print("The sticker is not here.")
        # If required, update the belief network to consider this last episode
        if withUpdate:
            new_data = []
            if self.mature:
                # Mature ToM
                if choice == "A" and found:
                    new_data = [1, 1, 1, 1]
                elif choice == "B" and found:
                    new_data = [0, 0, 0, 0]
                elif choice == "A" and not found:
                    new_data = [0, 0, 0, 1]
                elif choice == "B" and not found:
                    new_data = [1, 1, 1, 0]
            else:
                # Immature ToM
                if choice == "A":
                    new_data = [1, 1, 1, 1]
                else:
                    new_data = [0, 0, 0, 0]
            print(new_data, "kkkkkkk")
            new_episode = Episode(new_data, self.robot.get_and_inc_time())
            self.robot.beliefs[informer].update_belief(new_episode)
            # Add the symmetric espisode too (with the same time value)
            self.robot.beliefs[informer].update_belief(new_episode.generate_symmetric())
        # Finally, resets the eye color just in case an animation modified it
       

    # Belief Estimation Phase
    def belief_estimation(self, side):
        if self.simulation:
            self.relocate_sticker()
        # Recognizes the informer
        informer = self.user
        # Finds the sticker location
        #side = None
        #side = random.choice(["A", "B"]) 
        [informant_belief, informant_action] = self.robot.beliefs[informer].belief_estimation(side)
        
        print("I know the staus is on the " + self.translate_side(side) + ".")
        print("I believe you think the sticker is on the " + self.translate_side(informant_belief))
        print("I also believe you would point " + self.translate_side(informant_action) + " to me.")

    def translate_side(self, side):
        return "A" if side == "A" else "B"

    
    def calculate_trust():
        pass

# #for i in range(50):

#     #APITrust2("user1").start(False, False)
# for i in range(50):
#     x=random.choice([True, False])
#     y=random.choice([True, False])
#     APITrust2("user1").start(x, y)
