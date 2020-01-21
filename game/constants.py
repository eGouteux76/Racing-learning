import os
import pygame

pygame.font.init()
TRACK_GREY = (108,108,108,255)
START_POINT = pygame.math.Vector2(455,237)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0, 255) 
BLUE = (0, 0, 255, 255)
RED = (255, 0, 0, 255)
FPS = 60
CP_REWARD = 10 # reward when crossing a checkpoint
SCORE_DECAY = 1/1000 # score decay per millisecond in circuit
DIE_PENALTY = 100
N_TOURS = 1
screen_size = (1024,768)
sep = os.path.sep
MAP_PATH = os.path.split(os.getcwd())[0] + sep +"resources"+sep

font = pygame.font.SysFont("comicsansms", 40)

def load_circuit():
	circuit = pygame.image.load(MAP_PATH).convert_alpha()
	return circuit

class Config:
	"""
	This class contains all the constants for the car and the world, for interesting results,
	some constants could be modified or randomized
	"""
	def __init__(self):
		self.gravity = 9.81 # m/s^2
		self.mass = 1000.0 # kg
		self.inertiaScale = 1.0 # multiply by mass for inertia
		self.halfWidth = 0.8 # centre to side of chassis (metres)
		self.cgToFront = 2.0 # centre of gravity to front of chassis (metres)
		self.cgToRear = 2.0 # centre of gravity to rear of chassis (metres)
		self.cgToFrontAxle = 1.25 # centre gravity to front axle
		self.cgToRearAxle = 1.25 # centre gravity to rear axle
		self.cgHeight = 0.55 # centre gravity height
		self.wheelRadius = 0.3 # includes tire and represents height of axle
		# self.wheelWidth = 0.2 # for render only
		self.tireGrip = 2. # How much grip tires have
		self.lockGrip = 0.9 # % of grip available when wheel is locked
		self.engineForce = 12000.
		self.brakeForce = 14000.
		self.eBrakeForce = self.brakeForce / 2.5
		self.weightTransfer = 0.2 # How much weight is transferred during acceleration/braking
		self.maxSteer = 3.14 # maximum steering angle in radians
		self.cornerStiffnessFront = 5.
		self.cornerStiffnessRear = 5.2
		self.airResist = 2.5 # air resistance ( * vel )
		self.rollResist = 8. # rolling resistance force ( * vel )
		self.safeSteer = False
		self.smoothSteer = True
		self.image_path = os.path.split(os.getcwd())[0] + sep+"resources"+sep+"car.png"
