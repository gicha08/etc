import threading
import time
import random

# The number of neurons are subject to change.
# The number of connections between neurons are subject to change
# There could be irregular tasks that are performed sometimes and not sometimes
# With all the above assumptions, the time to process all units will be different each time
# For example
# time0 = time.clock()
# for unit in all_units:
#		unit.run()
# system_update() # change in terms of number of neurons, connections, etc. may happen once in a long while
# time1	= time.clock()
# --> 'time1 - time0' will be different each time
# 		i.e. for t=0,1,2,3,..., the gap between each time step will be different.
#			This is a problem in real time pattern recognition such as audio, motion
#			Need for timer and synchronization

def cal_dist(a,b):
	s = 0
	for i in range(len(a)):
		s += (a[i]-b[i])**2
	return s**0.5

# a = 0.02; b = 0.2; c = -65; d = 8;  % RS (Regular Spiking)
# a = 0.1; b = 0.2; c = -65; d = 2;  % FS (Fast Spiking)
# a = 0.02; b = 0.2; c = -50; d = 2;  % CH (Chattering)
spikes = {}
spikes['RS'] = (0.02,0.2,-65,8)
spikes['FS'] = (0.1,0.2,-65,2)
spikes['CH'] = (0.02,0.2,-50,2)
	
# Currently, updating connection weights which happens in each synapse when its neuron calls is done
# Initial state and change of network connectivity is yet to be designed and implemented
class Neuron:	
	def __init__(self, coords, spike_type='RS'):
		'''
		The information about neighboring neurons are to be obtain through its synapses and axons.
		Each synapse has a connection to an axon which belongs to another neuron
		'''
	
		# Center of the neuron
		self.coords = coords
		
		# Input : list of synapses
		self.synapses = []
		# Output : list of axons
		self.axons = [] 
			
		# spike variable initialize
		self.spike_type = spike_type
		a, b, c, d = spikes[self.spike_type]
		self.v = c
		self.u = b * c

	def __str__(self):
		'''Print Neuron Information.'''
		return "%s : %s %s" % (str(self.coords),str(self.synapses),str(self.axons))
	
	def get_coords(self):
		return self.coords
		
	def spike(self,v,u,I):
		a, b, c, d = spikes[self.spike_type]
		if v >= FIRING:
			v = c
			u = u + d
		v =  v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)
		u = u + a*(b*v - u)
		return v, u
		
	def add_synapse(self,sy):
		self.synapses.append(sy)
		
	def add_axon(self,ax):
		self.axons.append(ax)
		
	def run(self):
		# calculate input sum
		sum_input = 0
		for s in self.synapses:
			sum_input += s.get_input()
		
		# update spike variables
		self.v, self.u = self.spike(self.v,self.u,sum_input)
		
		# update axon
		for axon in self.axons:
			axon.feed(self.v)

		# update weight
		for synapse in self.synapses:
			synapse.update_weight()
		
SIG_MOV = 10
FIRING = 30
DECAY = 0.7
WEIGHT_RATE = 0.1

class Axon:
	def __init__(self, neuron):
		''' '''
		# The neuron to which this belongs
		self.neuron = neuron
		self.src = self.neuron.get_coords()	
		
		# each firing is represented by its distance from the origin
		self.firings = []
		self.on_fire = False
	
	def set_synapse(self, synapse):
		# The synapse to which this connects
		self.synapse = synapse
		self.dst = self.synapse.get_coords()

		# length in micro-meter
		self.length = cal_dist(self.src,self.dst)

	def feed(self,val):
		self.firings = [k+SIG_MOV for k in self.firings]	# move firings toward the synapse
		if val >= FIRING:
			self.firings.append(0)	# add new firing which is at distance 0 from the originator
		if self.firings and self.firings[0] > self.length:	# oldest one arrived at synapse. time interval should be short enough to have only one firing at synapse at most
			self.firings.pop(0)
			self.on_fire = True
		else:
			self.on_fire = False

class Synapse:
	def __init__(self, neuron):
		''' 
		A synapse has only one connection from an axon.
		Synapse methods (get_input(), update_weight()) are called by the neuron to which it belongs.
		'''
		# The neuron to which this belongs
		self.neuron = neuron
		
		# presynapse is later set by set_presynapse()
		# presynapse is supposed to be an axon from another neuron
		self.presynapse = None
		
		# Currently assume synapse is right next to the neuron
		self.coords = self.neuron.coords
			
		# Weight is determined and updated later by the interaction with the axon
		self.weight = 0
		
		# Initial input
		self.in_val = 0
	
	def set_presynapse(self, presynapse):
		self.presynapse = presynapse
		
	def get_coords(self):
		return self.coords
		
	def get_input(self,val):
		if self.presynapse:
			if self.presynapse.on_fire:
				self.in_val = 1
			else:
				self.in_val = DECAY * self.in_val
		else:
			self.in_val = 0
		return self.weight * self.in_val
	
	def update_weight(self):
		# Synapse Timing Dependent Plasticity
		if self.neuron.v > FIRING:
			#self.weight += WEIGHT_RATE * self.weight * self.in_val
			self.weight += WEIGHT_RATE * self.in_val
			self.v_trace = 1
		else:
			self.v_trace = DECAY * self.v_trace
			if self.presynapse.on_fire:	# presynapse spike after postsynapse spike
				#self.weight -= WEIGHT_RATE * self.weight * self.v_trace
				self.weight -= WEIGHT_RATE * self.v_trace

				
class NeuronSet:
	def __init__(self,num, dist="random"):
		'''
		For now, create neurons distributed in a cubic space
		'''
		self.neurons = []
		if dist=="uniform":
			# The number of neuron may not match the total number
			# eg) num = 10 will create 8 (2x2x2) neurons
			num_in_one_dim = int(num**(1.0/3))
			positions = [k/float(num_in_one_dim) for k in range(num_in_one_dim)]
			for i in positions:
				for j in positions:
					for k in positions:
						coords = (i,j,k)
						self.neurons.append(Neuron(coords))
		elif dist=="random":
			for i in range(num):
				coords = (random.random(),random.random(),random.random())
				self.neurons.append(Neuron(coords))
		else:
			print "Creating Neurons: Invalid Option", dist

	def show(self):
		'''Print neurons.'''
		for n in self.neurons:
			print n

	def connect(self,method="fully"):
		if method == "fully":
			for n1 in self.neurons:
				for n2 in self.neurons:
					if n1 == n2:	# avoid connection to itself
						continue
					# create a synapse
					sy = Synapse(n1)
					n1.add_synapse(sy)
					# create an axon
					ax = Axon(n2)
					n2.add_axon(ax)
					# connect the synapse and axon
					sy.set_presynapse(ax)
					ax.set_synapse(sy)
	
	
#n = Neuron((1.1,2.2,3.3))
#print n

neurons = NeuronSet(30,"random")
neurons.connect("fully")
neurons.show()

## Test for timer
#stop_flag = False
#def timer0():   
#	interval = 0.0001
#	base_time = time.clock()	# time.time() is not precise
#	cnt = 1
#	while True:
#		if stop_flag == True:
#			break
#		now = time.clock()
#		if now - base_time > interval*cnt:
#			cnt += 1
#			if cnt % int(0.1/interval) == 0:
#				print now
#		#time.sleep(interval/2)	# time.sleep() is not precise
#  	
#
#th = threading.Thread(target=timer0)
#
#th.start()
#time.sleep(5)
#stop_flag = True
#th.join()


     
