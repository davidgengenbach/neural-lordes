import numpy as np
import random


class connection:
	def __init__(self, target):
		self.target_index = target
		self.strength = 0.5
		self.was_also_firing_at_activation = False


class perceptron:
	def __init__(self, neighborhood):
		self.action_level = 0.5
		self.activation_threshold = 0.8

		self.desired_fire_rate = 0.5
		self.current_fire_rate = 0

		self.connections = [connection(i) for i in range(len(neighborhood))]
		self.is_firing = False

		self.neighborhood = neighborhood
		self.index = len(neighborhood)

		neighborhood.append(self)

	def decay_action(self):
		self.action_level = self.action_level * 0.9

	def receive_action(self, level):
		print "level", level
		self.action_level += level
			

	def check_fire(self):
		if self.action_level > self.activation_threshold:
			self.is_firing = True
			for i in self.connections:		
				i.was_also_firing_at_activation = self.neighborhood[i.target_index].is_firing
		else:
			self.is_firing = False

	def fire(self):
		for i in self.connections:
			self.neighborhood[i.target_index].receive_action(i.strength)
			if i.was_also_firing_at_activation:
				i.strength = i.strength * 0.9 #reduce connection strength, if other has fired before 
			else:
				i.strength = i.strength * 1.1 #otherwise increase

	def update(self):
		self.decay_action()
		self.check_fire()

		if self.is_firing:
			self.fire(self.neighborhood)


		
neighborshit = []

penis = perceptron(neighborshit)
penor = perceptron(neighborshit)

print penor.index
print penor.connections

print penor.action_level
print penis.action_level

penis.update()
penor.update()

print penor.action_level
print penis.action_level

penor.fire()

print penor.action_level
print penis.action_level

penor.fire()

print penor.action_level
print penis.action_level









#		db/dt + b - (de/dt + (1+ev) E)