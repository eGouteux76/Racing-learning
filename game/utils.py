import math
import pygame
import car
from constants import *


'''
retourne le point de collision et la distance de collision en fonction d'une direction
'''
def distanceToCollision(position, circuit, direction, checkDistance=1, precision=10, max_iter=10):
	# print(direction)
	startPos = pygame.Vector2(position.x, position.y)
	w,h = circuit.get_size()
	endPos = startPos
	while not collides(endPos, circuit):
		endPos += direction * checkDistance
		if endPos.x > w:
			endPos.x = w-1
		if endPos.y > h:
			endPos.y = h-1
		if endPos.x < 0:
			endPos.x = 0
		if endPos.y < 0:
			endPos.y = 0
	dist = lambda p1, p2: math.hypot(p1.x - p2.x, p1.y - p2.y)

	i = 0
	while (i < max_iter) and (dist(startPos, endPos) > precision):
		midPos = (endPos+startPos)/2

		if not collides(midPos, circuit):
			startPos = midPos
		else:
			endPos = midPos
		i += 1
	midPos = (endPos+startPos)/2
	# print(midPos)
	m = [int(midPos.x), int(midPos.y)]
	return m, dist(position, midPos)


'''
return false if the car is on the road, else return true.
'''
def collides(position, circuit, inColor=TRACK_GREY):
	return not circuit.get_at((round(position.x), round(position.y))) == inColor


def get_distances_and_draw (car, circuit, screen):
    
    vect_x = [0,0,0.87,0.5,0.87,0.5,1]
    vect_y = [-1,1,0.5,0.5,-0.5,-0.5,0]
    
    distances = []
    for i in range(len(vect_x)):
        m, d = distanceToCollision(car.position, circuit, pygame.Vector2(vect_x[i],vect_y[i]).rotate(math.degrees(car.heading)))
        pygame.draw.circle(screen, GREEN, m, 10)
        distances.append(d)
        
    return distances