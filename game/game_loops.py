import os
import pygame
from pygame.locals import *
import math
from constants import *
from car import Car
from circuit import Circuit
import utils

def agent_inputs(vectors, car, circuit_img):
    vectors_distance = [utils.distanceToCollision(car.position,
     circuit_img, vector.rotate(math.degrees(car.heading)))[1] for vector in vectors]
    car_data = [car.velocity_c[0], car.velocity_c[1], car.yawRate, car.accel_c[0], car.accel_c[1]]
    
    #on "normalise" les distances pour pas avoir trop de la merde au début
    vectors_distance = [i*0.01 for i in vectors_distance]

    return car_data + vectors_distance


def game_loop(screen, clock, car, vectors, circuit, is_ai=True, checkpoint=0, render=True):
    dtms = clock.tick(FPS)
    circuit_img = circuit.get_nth_checkpoint(checkpoint)
    score_update = 0
    if render:
        pygame.display.flip()
    if not is_ai:
        running = car.inputs.update()
    renderedText = car.getStats()
    text = font.render(renderedText, True, (0, 128, 0))


    car.update(dtms)
    

    surface = car.render()
    surface = pygame.transform.rotate(surface, -math.degrees(car.heading))
    rot_rect = surface.get_rect()
    rot_rect.move(car.position.x, car.position.y)
    if render:
        screen.fill(BLACK)
        screen.blit(circuit_img, (0,0))
        pygame.draw.rect(screen, GREEN, surface.get_rect())
        screen.blit(surface, (car.position.x, car.position.y))
        screen.blit(surface, car.position)
        screen.blit(text, (0,0))

    for vector in vectors:
        m, d = utils.distanceToCollision(car.position, circuit_img, vector.rotate(math.degrees(car.heading)))
        if render:
            pygame.draw.circle(screen, GREEN, m, 10)
    

    onCheck = utils.onCheckpoint(car.position, circuit_img)

    if onCheck:
        checkpoint += 1
        circuit_img = circuit.get_nth_checkpoint(checkpoint)
        score_update += CP_REWARD*(checkpoint+1)

    out_of_circuit = utils.collides(car.position, circuit_img)
    running = not (out_of_circuit or checkpoint == circuit.n_checkpoints * N_TOURS)

    # print(utils.collides(car.position, circuit_img))
    # print(car.position)
    if render:
        pygame.display.flip()
    score_update -= dtms*SCORE_DECAY + out_of_circuit * DIE_PENALTY
    return running, score_update, checkpoint