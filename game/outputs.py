import math
import car
from constants import circuit
from pygame.math import Vector2

class Outputs:


	def __init__(self, car_data_fields, sensor_vectors):

		"""
		car_data : liste de string des donnees a recuperer de la voiture (comme la vitesse etc)
		sensor_vectors : liste des directions des capteurs pour detecter la distance a la prochaine collision
		"""

		self.car_data_fields = car_data_fields
		self.sensor_vectors = sensor_vectors
		self.collision_distances = []
		self.car_outputs = []
		self.outputs = []
		self.score = 0

	def get_collision_distances(self, car):

		self.collision_distances = []

		for vect in sensor_vectors:
			rotated_vect = vect.rotate(math.degrees(car.heading))
			center, dist = utils.distanceToCollision(car.position, circuit, rotated_vect)
			self.collision_distances.append(dist)

	def get_car_data(self, car):

		self.car_outputs = []
		car_data = car.getCarDict()

		for key in self.car_data_fields:

			if key in car_data.keys:
				self.car_outputs.append(car_data[key])
			else:
				print("Key {} does not exist".format(key))

		




