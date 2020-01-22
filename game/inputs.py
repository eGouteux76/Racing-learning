import os
import pygame
from pygame.locals import *
import math

class Inputs:
    def __init__(self):
        pygame.key.set_repeat(1) # if any key is kept down, pygame will register it every 1ms
        self.left = 0.
        self.right = 0.
        self.throttle = 0.
        self.brake = 0.
        self.ebrake = 0.

    def reset(self):
        self.left = 0.
        self.right = 0.
        self.throttle = 0.
        self.brake = 0.
        self.ebrake = 0.
        
    #action :left, right, throttle, brake, ebrake 
    def add(self, liste):
        self.left = liste[0]
        self.right = liste[1]
        self.throttle = liste[2]
        self.brake = liste[3]
        self.ebrake = liste[4]
        
    def update(self):
        self.reset()
        key = pygame.key.get_pressed()

        if key[pygame.K_UP] != 0:
            self.throttle += 1.
        if key[pygame.K_DOWN] != 0:
            self.brake += 1.
        if key[pygame.K_LEFT] != 0:
            self.left += 1.
        if key[pygame.K_RIGHT] != 0:
            self.right += 1.
        if key[pygame.K_SPACE] != 0:
            self.ebrake += 1.

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Stop")
                return False
        return True

    def display(self):
        print("Left = {}\nRight = {}\nThrottle = {}\nBrake = {}\nEBrake = {}"
            .format(self.left, self.right, self.throttle, self.brake, self.ebrake))



