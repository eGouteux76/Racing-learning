import pygame
from pygame.locals import *
from constants import START_POINT, screen_size
from car import Car
from circuit import Circuit
from game_loops import game_loop, agent_inputs
import model
import numpy as np
from Heuristique import heuristic, params
#TODO gestion score par voiture
from game import Game


def net_to_input(net):
    inp = Inputs()
    inp.left = net[0]
    inp.right = net[1]    
    inp.throttle = net[2]
    inp.brake = net[3]
    return inp


pygame.init()
screen = pygame.display.set_mode(screen_size)
circuit = Circuit()
circuit_img = circuit.images[0] # pour tester les collisions, le checkpoint n'est pas important
circuit.display()
clock = pygame.time.Clock()
car = Car(0.,START_POINT)
running = True
vectors = [pygame.Vector2(0.,-1),pygame.Vector2(1,-1),pygame.Vector2(1,-0.5),pygame.Vector2(1,0.),pygame.Vector2(1,0.5),pygame.Vector2(1,1),pygame.Vector2(0.,1)]
score = 0
checkpoint = 0


path = "models/"
reload_model = True
reload_nb_tentative = True
INPUT_LEN = 12

is_renforcement = True
is_heuristique = False
is_ai = True

train = True
agent = None



if is_renforcement:
    agent = model.reload_agent_renforcement(path,reload_model,reload_nb_tentative)


if reload_model and is_renforcement:
    
    nb_tentative = model.get_nb_tentative(path)
    agent = model.Agent(INPUT_LEN,4,nb_tentative)
    agent.load_model(path + "DDQN_model_cp.h5")
else : 
    agent = model.Agent(INPUT_LEN,4)
        
max_frame = 1200

t = 0

while train:
    
    t = t+1
    pygame.event.get()
    running, score_update, checkpoint = game_loop(screen, clock, car, vectors, circuit, is_ai=is_ai,
        checkpoint=checkpoint, render=True)
        
    if score_update >0 :
        t=0
    if t > max_frame :
        score_update -= 50
        running = False
        
    if is_renforcement:
            
        network_inputs = agent_inputs(vectors, car, circuit_img)
        car_inputs, active = agent.step(network_inputs, score_update, running) #score à changer
        # agent_decision retourne une classe Inputs,  
        # avec les entrees que l'IA a decide de faire. 
        if active :
            car.inputs.add(car_inputs) # actualisation des inputs de la voiture par l'ia
        
    if is_heuristique :
        network_inputs = agent_inputs(vectors, car, circuit_img)
        print("netw",network_inputs)
        car_inputs = heuristic(network_inputs,params)
        print("input",car_inputs)
        if running:
            car.inputs.add(car_inputs)
                
    score += score_update
    print("Score = ", score)
        
    if not running :     
        checkpoint = 0
        car = Car(0.,START_POINT)
        running = True
        score = 0
        t=0
        print("restart")
    
        
