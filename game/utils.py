import math
import pygame
import car
from constants import *

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

def collides(position, circuit, inCircuitColors=[TRACK_GREY, RED, GREEN]):
	return not circuit.get_at((int(round(position.x)), int(round(position.y)))) in inCircuitColors

def onCheckpoint(position, circuit, checkpointColor=GREEN):
	# print(circuit.get_at((int(round(position.x)), int(round(position.y)))))
	return circuit.get_at((int(round(position.x)), int(round(position.y)))) == checkpointColor

