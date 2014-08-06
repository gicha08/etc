from bfs import *

# actions
actions = ((0,1), (0,2), (1,0), (1,2), (2,0), (2,1)) # (src peg, target peg)

# Intial state
ndisk = 22
start_state = [range(ndisk,0,-1), [], []]
	
class HanoiSearch(BFS):
		
	def is_legal_action(self, state, action):
		#print state, action
		src, dest = action 
		return  state[src] != [] and  state[src][-1] <= self.ndisk and (state[dest] == [] or state[dest][-1] > state[src][-1])
			
	def set_target(self, ndisk, target_peg):
		self.ndisk  = ndisk
		self.target_peg = target_peg
		
	def is_goal(self, state):
		'''
		Each state is composed of three stacks(pegs) with disks represened by numbers corresponding their sizes
		'''
		if len(state[self.target_peg]) < self.ndisk:
			return False
		for i in range(1, self.ndisk+1):
			if state[self.target_peg][-i] != i:
				return False
		return True
		
	def perform_action(self, state, action):
		src, dest = action
		new_state = [state[0][:], state[1][:], state[2][:]]
		new_state[dest].append(new_state[src].pop())
					
		return new_state
						 
	def successors(self, state):
		successors = []
		for a in actions:
			if self.is_legal_action(state,a):
				new_state = self.perform_action(state, a)
				successors.append((a,new_state))
#		print 'successors of', state, '-->', successors			
		return successors
				
		
	
	
#hanoi = HanoiSearch(start_state)
#hanoi.set_target(ndisk,target_peg)
#
#if hanoi.search():
#	print hanoi.nid, hanoi.path
#else:
#	print "No Solution !"


def get_src_peg(state):
	for i, peg in enumerate(state):
		if (peg != [] and peg[-1] == 1):
			return i
	return None

def is_goal(state,ndisk,target_peg):
	'''
	Each state is composed of three stacks(pegs) with disks represened by numbers corresponding their sizes
	'''
	if len(state[target_peg]) < ndisk:
		return False
	for i in range(1, ndisk+1):
		if state[target_peg][-i] != i:
			return False
	return True

	
def perform_action(state, action):
	src, dest = action
#	new_state = [state[0][:], state[1][:], state[2][:]]
#	print state, action
	state[dest].append(state[src].pop())
				

def hanoi_puzzle(state, ndisk, src_peg, target_peg, buffer_peg):
	if ndisk == 0:
		return []
	else:
		actions1 = hanoi_puzzle(state, ndisk-1, src_peg, buffer_peg, target_peg)
		action2 = (src_peg,target_peg)
		perform_action(state,action2)

#		actions3 = hanoi_puzzle(state, ndisk-1, buffer_peg, target_peg, src_peg)
		
		changes = {buffer_peg:target_peg, target_peg:src_peg, src_peg:buffer_peg}
		print 'number of actions1', len(actions1)
		actions3 = [(changes[s],changes[d]) for s,d in actions1]
		for act in actions3:
			perform_action(state,act)
			
		return actions1 + [action2] + actions3
		

hanoi_puzzle(start_state, ndisk, 0, 2, 1)
print start_state