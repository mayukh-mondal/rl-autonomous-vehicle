import GameEnv
import pygame
import numpy as np

from ddqn_keras import DDQNAgent

from collections import deque
import random, math


#Setting the total game time, number of episodes and the learning rate
TOTAL_GAMETIME = 10000
N_EPISODES = 10000
REPLACE_TARGET = 10

#Instantiate the game environment with a required fps
game = GameEnv.RacingEnv()
game.fps = 120

#Initialize the game time , game history(to store the game history)
GameTime = 0 
GameHistory = []
renderFlag = True

#Initialize the game agent with our required values
ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=0.02, epsilon_end=0.01, epsilon_dec=0.999, replace_target= REPLACE_TARGET, batch_size=64, input_dims=19,fname='ddqn_model.h5')

#Loading up the model into our agent and giving it the parameters
ddqn_agent.load_model()
ddqn_agent.update_network_parameters()

ddqn_scores = []
eps_history = []


def run():

    for e in range(N_EPISODES):
        #reset env 
        game.reset()

        #Done : tells us if the car has completed the course or not.
        #Score : tells us the score of the car
        #Counter : counts the number of attempts so far.
        #Gtime : tells us the global time of the environment
        done = False
        score = 0
        counter = 0
        gtime = 0

        #first step
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        while not done:
            
            #This part is to stop the game when required.
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    run = False
                    return

            #new
            #Make the agent take a choice on what to do.
            action = ddqn_agent.choose_action(observation)
            #Calculate what happens once the action has been taken
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            if reward == 0:
                counter += 1
                #if the counter reaches 100, our car has completed the course
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            observation = observation_

            #We increase the gtime in every frame . this is to give the car points for survival.
            #This stops our agent from crashing into the wall immediately.
            gtime += 1

            #However, this can couse the car to simply stand still and farm points just for existing.
            #Terminate the game after a set max time.
            if gtime >= TOTAL_GAMETIME:
                done = True

            # if renderFlag:
                # game.render(action)
            game.render(action)



run()        
