import sys

class Node:
	def __init__(self, state, parent=None, action= None):
		self.state = state
		self.parent = parent
		self.action = action

class BFS:
  
	def __init__(self, state):
		# initialize with the root node
		self.fringe = [Node(state)]
		self.nid = 1	# increment by one with each new node generated
   
	# Performs search until a goal is reached
	def search(self):
		while( self.fringe ):
			res = self.search_step()
			if res:
				return True
		return False
  
  # Performs a single iteration of search
	def search_step(self):

		n = self.fringe.pop(0);	# pop the oldest one --> Breadth First Search
		successors = self.successors(n.state)

		for action, succ in successors:
			#print 'successor',succ.playerCoord,succ.objects	# for debugging
			# succ is a Sokoban state, not a node yet
			if self.is_goal(succ):
				self.path = [succ]
				while n:
					self.path.append(n.state)
					n = n.parent
				self.path.reverse()
				return True
			else:
				self.fringe.append(Node(succ,n,action))
				self.nid += 1

		return False
  
 	  	 		
	#
	# The followings must be overloaded by the subclass
	#
	def is_goal(self,state):
		return
	
	def successors(self,state):
		return

