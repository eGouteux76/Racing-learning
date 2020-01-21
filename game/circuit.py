import pygame
from constants import MAP_PATH
class Circuit:
	
	def __init__(self, directory=MAP_PATH, name='map', img_format='.png', n_checkpoints=14):
		path = directory + name
		self.images = [pygame.image.load(path+str(i)+img_format).convert_alpha() for i in range(n_checkpoints)]
		self.n_checkpoints = n_checkpoints

	def get_nth_checkpoint(self, n):
		return self.images[n % self.n_checkpoints]

	def display(self):
		print(self.images)

if __name__ == '__main__':
	pygame.init()
	c = Circuit()
	c.display()



