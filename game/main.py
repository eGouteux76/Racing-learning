import pygame
from pygame.locals import *
from constants import START_POINT, screen_size
from car import Car
from circuit import Circuit
from game_loops import game_loop, agent_inputs
import model
import numpy as np
#TODO gestion score par voiture

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
reload_model = False
model_path = "models/DDQN_model_cp.h5"

INPUT_LEN = 11

is_ai = True
train = True

agent = model.Agent(INPUT_LEN,6)


if reload_model:
    agent.load_model(model_path)
    

while train:
    
    running, score_update, checkpoint = game_loop(screen, clock, car, vectors, circuit, is_ai=is_ai,
     checkpoint=checkpoint, render=True)
    
    if is_ai:
        
        network_inputs = agent_inputs(vectors, car, circuit_img)
        car_inputs, active = agent.step(network_inputs, score_update, running) #score à changer
        # agent_decision retourne une classe Inputs,  
        # avec les entrees que l'IA a decide de faire. 
        if not np.array_equal(car_inputs,agent.no_action) :
            car.inputs.add(car_inputs) # actualisation des inputs de la voiture par l'ia
        
    score += score_update
    print("Score = ", score)
    
    if not running : 
        agent.nb_tentative += 1           
        checkpoint = 0
        car = Car(0.,START_POINT)
        running = True
        score = 0
        print("restart")
    
    
    
