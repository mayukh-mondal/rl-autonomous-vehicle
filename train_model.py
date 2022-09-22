import GameEnv
import pygame
import numpy as np
from ddqn_keras import DDQNAgent

TOTAL_GAMETIME = 1000 # Max game time for one episode
N_EPISODES = 4000
REPLACE_TARGET = 50 

#import the game environment
game = GameEnv.RacingEnv()
game.fps = 120

GameTime = 0 
GameHistory = []
renderFlag = False

ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995, replace_target= REPLACE_TARGET, batch_size=512, input_dims=19)

# if retraining existing model
ddqn_agent.load_model()

#stores the history of scores and epsilons to train the model every few epochs
ddqn_scores = []
eps_history = []

def run():

    for e in range(N_EPISODES):
        
        game.reset() #reset env 

        done = False
        score = 0
        counter = 0

        #updating the model        
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        gtime = 0 # set game time back to 0
        
        renderFlag = False # if you want to render every episode set to true

        if e % 1 == 0 and e > 0: # render every 10 episodes
            renderFlag = True

        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    return
            
            #step function
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            #if car is inactive then quit after 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True 
            else:
                counter = 0
            
            #update score
            score += reward

            #stores the values
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()
            
            #update global uptime for the current episode
            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            #sneak-peek every 10 episodes or so
            if renderFlag:
                game.render(action)

        #store the records of performance metrics for model updation
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])

        #if time to update model, then do so
        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()

        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("save model")
            
        #print out stats on the terminal for gross evaluation as the training occurs.
        print('Episode: ', e,'Score: %.2f' % score,
              ' Average Score %.2f' % avg_score,
              ' Current epsilon value: ', ddqn_agent.epsilon)   

run()        
