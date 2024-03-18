import numpy as np
import matplotlib.pyplot as plt

#Building the environment
#Intialization

Delta = 0.005
Gamma = 0.9
Noise = 0.2

states =[]
for i in range (3):
    for j in range (4):
        states.append((i,j))

rewards ={}
for i in  (states):
    rewards[i] = 0
    if i == (0,3):
       rewards[i] = 1

    elif i == (1,3):
        rewards[i] = -1

    elif i == (1,1):
        rewards[i] = "WALL"

    


#Possible action definition
actions = {
(0,0):("Down", "Right"),
(0,1):("Right", "Left"),
(0,2):("Right", "Left","Down"),
(0,3):("Left","Down"),
(1,0):("Up", "Down"),
(1,2):("Up","Down","Right"),
(1,3):("Down", "Left"),
(2,0):("Up","Right"),
(2,1):("Left","Right"),
(2,2):("Up", "Right","Left"),
(2,3):("Up","Left")


}           
#Policy definition
policy ={}

for s in actions.keys():
    policy[s] = np.random.choice(actions[s])


#Value function definition
    Values = {}
for s in states:
    Values[s] = 0

Values[(0, 3)] = 1
Values[(1, 3)] = -1
Values[(1, 1)] = "WALL"


#Value iteration
iteration = 0 

while True :
    maxium_change = 0

    for s in states:
         if s in policy:
             old_v = Values[s]
             new_v = 0

             for a in actions[s]:
                 if a == "Up" :
                    nxt = [s[0]-1, s[1]]
                 if a == "Down" :
                    nxt =[s[0]+1,s[1]]
                 if a == "Right" :
                     nxt= [s[0],s[1]+1]
                 if a == "Left":
                     nxt= [s[0],s[1]-1]  


#Choose a new random action to do (transition probability)
    random_1 =np.random.choice([i for i in actions[s] if i != a])
    if random_1 == 'Up':
        action = [s[0]-1, s[1]]
    if random_1 == 'Down':
        action = [s[0]+1, s[1]]
    if random_1 == 'Left':
       action = [s[0], s[1]-1]
    if random_1 == 'Right':
       action = [s[0], s[1]+1]

# Caclculating the value    
    nxt = tuple(nxt)
    action = tuple(action)
    v = rewards[s] + (Gamma * ((1-Noise)* Values[nxt]+(Noise * Values[action])))
    if v > new_v :
      new_v= v
    policy[s] = action
                  
            
    Values[s] = new_v
    maxium_change = max(maxium_change, np.abs(old_v - Values[s]))

    if maxium_change < Delta:
        break
    iteration += 1



    



                 
                 




