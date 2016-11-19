#import numpy as np
import random, math




class output_connectron:
	def __init__(self):
		self.mean = 0
		self.threshold = 0.1
		self.input_weights = []

	def activate(self, inputs, supervision):
		self.mean = supervision

		while(len(inputs) > len(self.input_weights)):
			self.input_weights.append(random.randint(1,100)/100)

		input_sum = 0
		for i, value in enumerate(inputs):
			input_sum += value*self.input_weights[i]
		input_mean = input_sum / len(inputs)
		
		if(input_sum < self.threshold):
			return 0	

		diff = abs(input_sum) - abs(self.mean)

		if(diff > 0):
			for i, val in enumerate(inputs):
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					self.input_weights[i] -= 0.02 * math.copysign(1, self.input_weights[i])
				else:
					self.input_weights[i] *= 0.99
		else:
			for i, val in enumerate(inputs):
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					self.input_weights[i] += 0.02 * math.copysign(1, self.input_weights[i])
				else:
					self.input_weights[i] *= 0.99

		return input_sum



class connectron:
	def __init__(self):
		self.mean = 0
		self.threshold = 0.1

		self.input_weights = []

	def activate(self, inputs):
		#if(len(self.input_weights) < 1):
		#	self.input_weights = [0.02, 0.71]

		while(len(inputs) > len(self.input_weights)):
			self.input_weights.append(random.randint(20,100)/100)

		#weight_mean = 0
		#for i in self.input_weights:
		#	weight_mean += i
		#weight_mean /= len(self.input_weights)

		input_sum = 0
		for i, value in enumerate(inputs):
			input_sum += value*self.input_weights[i]
		input_mean = input_sum / len(inputs)
		

		if(input_sum < self.threshold):
			return 0

		self.mean = 1 #comment this out if it should learn mean
		if(self.mean < self.threshold):
			self.mean = input_sum
			return 0

		diff = abs(input_sum) - abs(self.mean)
		#print("diff:", diff)

		if(diff > 0):
			# overshoot case
			#print("overshoot")
			#self.mean *= 1.01

			for i, val in enumerate(inputs):
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					#print(abs(i) , abs(input_mean))
					self.input_weights[i] -= 0.02 #* math.copysign(1, self.input_weights[i])
				else:
					self.input_weights[i] *= 0.99
		else:
			# undershoot case
			#print("undershoot")
			#self.mean *= 0.99
			
			for i, val in enumerate(inputs):
				#print(val, self.input_weights[i])
				#print(abs(val*self.input_weights[i]), ">", abs(input_mean))
				if(abs(val*self.input_weights[i]) > abs(input_mean)):
					self.input_weights[i] += 0.02 #* math.copysign(1, self.input_weights[i])
				else:
					self.input_weights[i] *= 0.99

		return input_sum

def test_run(iterations):
	a = output_connectron()	
	b = output_connectron()	

	input_vector_a = [0, 1, 0, 1]
	input_vector_b = [1, 0, 1, 0]

	for i in range(iterations):
		rand_vector = [random.randrange(0,1), random.randrange(0,1), random.randrange(0,1), random.randrange(0,1)]

		activation = b.activate(rand_vector, 0)
		activation = a.activate(rand_vector, 0)
		activation = b.activate(input_vector_a, 0)
		activation = a.activate(input_vector_a, 1)
		activation = b.activate(input_vector_b, 1)
		activation = a.activate(input_vector_b, 0)
		#print(activation, a.input_weights)

	print(a.activate(input_vector_a, 0))
	print(a.activate(input_vector_b, 0))
	print(b.activate(input_vector_a, 0))
	print(b.activate(input_vector_b, 0))

test_run(1000)

