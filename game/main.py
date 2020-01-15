import os
import pygame
from pygame.locals import *
import math
from constants import *
from car import Car
import utils


pygame.init()
screen = pygame.display.set_mode(screen_size)
circuit = load_circuit()

clock = pygame.time.Clock()
car = Car(0.,START_POINT)
running = True
while running:
    dtms = clock.tick(FPS)
    pygame.display.flip()
    running = car.inputs.update()
    renderedText = car.getStats()
    text = font.render(renderedText, True, (0, 128, 0))
    car.update(dtms)
    surface = car.render()
    surface = pygame.transform.rotate(surface, -math.degrees(car.heading))
    rot_rect = surface.get_rect()
    rot_rect.move(car.position.x, car.position.y)
    # surface_2, rect = rotate(surface, car.heading, pivot, offset)
    screen.fill(BLACK)
    screen.blit(circuit, (0,0))
    pygame.draw.rect(screen, GREEN, surface.get_rect())
    screen.blit(surface, (car.position.x, car.position.y))
    screen.blit(surface, car.position)
    
    #les distances sont récupérées de l'aile gauche à l'aile droite de la voiture
    distances = utils.get_distances_and_draw(car, circuit, screen) 
    screen.blit(text, (0,0))
    print(utils.collides(car.position, circuit))
    print(car.position)
    pygame.display.flip()
    
        