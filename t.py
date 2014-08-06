import random

class Stream:
	def __init__(self,data):
		self.idx = 0
		self.data = data
	    
	def HasNext(self):
		return (self.idx < len(self.data))
	def GetNext(self):
		val = self.data[self.idx]
		self.idx += 1
		return val
    
def get_random_object(stream):
	val = stream.GetNext()
	size = 1
	while stream.HasNext():
		newval = stream.GetNext()
		if random.randrange(size+1) == 0:
			val = newval
		size += 1
#		print size
	return val

class Line:
	def __init__(self,x1,y1,x2,y2):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.a = y2-y1
		self.b = x1-x2
		self.c = (x2-x1)*y1 - (y2-y1)*x1
	
	def value(self,x1,y1):
		return self.a*x1 + self.b*y1 + self.c
	def is_intersect(self,line2):
		if (self.value(line2.x1,line2.y1) * self.value(line2.x2,line2.y2) < 0) \
		and (line2.value(self.x1,self.y1) * line2.value(self.x2,self.y2) < 0):
			return True
		return False

class Smart_stack:
	def __init__(self):
		self.size = 0
		self.m = None
		self.data = []
		
	def push(self,data):
		if self.m:
			self.m = min(self.m,data)
		else:
			self.m = data
		self.data.append((data,self.m))
		self.size += 1
		
	def pop(self):
		val = self.data.pop()[0]
		self.m = self.data[-1][1]
		self.size -= 1
		return val
	
	def get_min(self):
		return self.m
	    
	def show(self):
		print self.data

class Tree:
	def __init__(self,val,left=None,right=None):
		self.val = val
		self.ltr = left
		self.rtr = right
		self.right_neighbor = None
	
	def show(self):
		if self.ltr == None or self.rtr == None:
			return [self.val, self.ltr, self.rtr, self.right_neighbor]
		else:
			return [self.val, self.ltr.show(), self.rtr.show(), self.right_neighbor]

def get_trees(tr,depth):
	if depth == 0:
		return [tr]
	else:
		return get_trees(tr.ltr,depth-1) + get_trees(tr.rtr,depth-1)

def add_nodes(tr, nodelists, level):
	if tr is not None:
		if level >= len(nodelists):
			nodelists.append([])
		nodelists[level].append(tr.val)
		add_nodes(tr.ltr, nodelists, level+1)
		add_nodes(tr.rtr, nodelists, level+1)

def get_nodes_at(tr,depth):
	if tr == None:
		return []
	elif depth == 0:
		return [tr.val]
	else:
		return get_nodes_at(tr.ltr, depth-1) + get_nodes_at(tr.rtr, depth-1)
		
def get_nodes_per_level(tr):
	
	if tr == None:
		return []
	else:
		nodelists = [[tr.val]]
		add_nodes(tr.ltr,nodelists,1)
		add_nodes(tr.rtr,nodelists,1)
	return nodelists	
	'''
	nodelists = []
	for i in range(height(tr)):
		nodelists.append(get_nodes_at(tr,i))
	return nodelists
	'''
	
def height(tr):
	if tr == None:
		return 0
	else:
		return max(height(tr.ltr), height(tr.rtr)) + 1

def add_neighbors(trlist):
	for i in range(len(trlist)-1):
		trlist[i].right_neighbor = trlist[i+1]
		
		
def add_right_neighbor(tr):
	h = height(tr)
	for i in range(1,h):
		q = get_trees(tr,i)
		add_neighbors(q)
		
def height_balance(tr):
	if tr == None:
		return (0, True)
	else:
		lval = height_balance(tr.ltr) 			
		rval = height_balance(tr.rtr)
		#print lval, rval
		return (max(lval[0],rval[0])+1, lval[1] and rval[1] and (abs(lval[0]-rval[0]) < 2))
		
def is_balanced(tr):
	return height_balance(tr)[1]
	
def is_binary_search_tree_helper(tr):
	if tr.ltr == None and tr.rtr == None:
		return (tr.val, tr.val)
	elif tr.ltr == None:
		right = is_binary_search_tree_helper(tr.rtr)
		return right and tr.val <= right[0] and (tr.val, right[1])
	elif tr.rtr == None:
		left = is_binary_search_tree_helper(tr.ltr)
		return left and left[1] <= tr.valand (left[0], tr.val)
	else: 
		left =	is_binary_search_tree_helper(tr.ltr)
		right =	is_binary_search_tree_helper(tr.rtr)
		#print left, right
		return left and right and left[1] <= tr.val and tr.val <= right[0] and (left[0], right[1])
		
def is_binary_search_tree(tr):
	return is_binary_search_tree_helper(tr) and True
					
def has_duplicate_char(ls):
	seen = {}
	for i in ls:
		if i in seen:
			return True 
		seen[i] = seen.get(i,0) + 1
	return False
		
def quicksort_range(ls,start_idx,end_idx): # end_idx is not included
#	print ls, start_idx, end_idx
	if start_idx + 1 >= end_idx:	# need at least two things to sort
		print 'return'
		return
		
	pivot_idx = start_idx
	for i in range(start_idx+1,end_idx):
		if (ls[i] < ls[pivot_idx]):
			val = ls[i]
			for k in range(i,pivot_idx,-1):
				ls[k] = ls[k-1]
			ls[pivot_idx] = val
			pivot_idx += 1
		# print i, pivot_idx, ls
	quicksort_range(ls,start_idx,pivot_idx)
	quicksort_range(ls,pivot_idx+1,end_idx)
		

def quicksort_inplace(ls):
	quicksort_range(ls,0,len(ls))
		
def compress_string(s):
	letter = s[0]
	cnt = 1
	ans = []
	for i in range(1,len(s)):
		if letter == s[i]:
			cnt += 1
		else:
			ans.append(letter + str(cnt))
			letter = s[i]
			cnt = 1
	ans.append(letter + str(cnt)) 
	if len(ans) < len(s):
		return "".join (ans)
	else:
		return s
		
class MyQueue:
	def __init__(self):
		self.stack1 = []
		self.stack2 = []
		
	def push(self,data):
		self.stack1.append(data)
	
	def pop(self):
		while self.stack1:
			self.stack2.append(self.stack1.pop())
		val = self.stack2.pop()
		while self.stack2:
			self.stack1.append(self.stack2.pop())
		return val

def pick_random_from_stream():
	cnt = 0
	selection = []
	number = raw_input("Enter a new integer ('q' to quit):")
	while number != 'q':
		if (len(selection) < 10):
			selection.append(int(number))
		else:
			if random.randrange(cnt) < 10:
				selection.pop(random.randrange(10))
				selection.append(int(number))
			
		cnt += 1
		number = raw_input("Enter a new integer ('q' to quit):")
		
	return selection

def stair_hops(n):
	if n <= 0:
		return 1
	elif n == 1:
		return 1
	elif n == 2:
		return 2
	else: #n >= 3:
		return stair_hops(n-1) + stair_hops(n-2) + stair_hops(n-3)

def	stair_hops_dp(n):
	hops = [1,1,2]
	for i in range(3,n+1):
		hops.append(hops[i-1]+hops[i-2]+hops[i-3])
	return hops[n]
	
def fibonacci(n):
	if n <= 1:
		return 1
	else:
		return fibonacci(n-1) + fibonacci(n-2)
		
def fibonacci_dp(n):
	fib = [1,1]
	for i in range(2,n+1):
		fib.append(fib[i-1] + fib[i-2])
	return fib[n]
	
def fibonacci_dp2(n):
	a = 1
	b = 1
	for i in range(2,n+1):
		c = a + b
		a = b
		b = c
	return c

def num_shortest_paths(x,y):
	if x == 0:
		return 1
	elif y == 0:
		return 1
	else:
		return num_shortest_paths(x-1, y) + num_shortest_paths(x,y-1)

def num_shortest_paths_dp(x,y):
	mat = [[1] * (x+1) for i in range(y+1)]
	#print mat	
	for xi in range(1,x+1):
		for yi in range(1,y+1):
			#print xi, yi
			mat[yi][xi] = mat[yi-1][xi] + mat[yi][xi-1]
			#print mat
	return mat[y][x]
		
def exponents2num(exps):
	return 3**exps[0] * 5**exps[1] * 7**exps[2]
	
def get_multiple_primes(k):
	q = [[0,0,0]]
	read_idx = 0
	for i in range(1,k):
		n = q[0][:]
		n[read_idx] += 1
		while exponents2num(n) <= exponents2num(q[-1]):
			if read_idx == 2:
				read_idx = 0
				q.pop(0)
			else:
				read_idx += 1
#			print n
			n = q[0][:]
			n[read_idx] += 1
			
		q.append(n)
		
	return q[-1]
	#return exponents2num(q[-1])
		
def num2set(n,ls):
	s = []
	for i, item in enumerate(ls):
		if n & (1 << i) != 0:
			s.append(item)
	return s
			
					
def get_subsets(ls):
	if ls == []:
		return [[]]
	else:
		subsets = get_subsets(ls[1:])
		return subsets + [[ls[0]]+s for s in subsets]

def get_subsets2(ls):
	sets = []
	for i in range(2**len(ls)):
		sets.append(num2set(i,ls))
	return sets
	
def get_permutations(s):
	if len(s) == 1:
		return [s]
	else:
		permuts = []
		for i in range(len(s)):
			c = s[i]
			rest = s[:i] + s[i+1:]
			#print c, rest
			sub_permuts = get_permutations(rest)
			permuts += [c+w for w in sub_permuts]
	return permuts
			
def get_parens(n):
	if n == 1:
		return ['()']
	else:
		parens = get_parens(n-1)
		new_parens = set([])
		for p in parens:
			new_parens.add('('+p+')')
			new_parens.add( p+'()')
			new_parens.add('()'+p)
	return new_parens
		
def pay_with_coins(n):
#	if n < 0:
#		return 0
#	elif n == 1:
#		return 1
#	else:
#		return pay_with_coins(n-25) + pay_with_coins(n-10) + pay_with_coins(n-5) + pay_with_coins(n-1)
	if n < 0:
		return []
	elif n == 0:
		return [[]]
	elif n == 1:
		return [[1]]
	elif n >= 25:
		return [[25]+p for p in pay_with_coins(n-25)]\
		 + [[10]+p for p in pay_with_coins(n-10) if 25 not in p] \
		 + [[5]+p for p in pay_with_coins(n-5) if (10 not in p) and (25 not in p) ] \
		 + [[1]+p for p in pay_with_coins(n-1) if (5 not in p) and (10 not in p) and (25 not in p)]
	elif n >= 10:
		return [[10]+p for p in pay_with_coins(n-10) if 25 not in p] \
		 + [[5]+p for p in pay_with_coins(n-5) if (10 not in p) and (25 not in p) ] \
		 + [[1]+p for p in pay_with_coins(n-1) if (5 not in p) and (10 not in p) and (25 not in p)]
	elif n >= 5:
		return [[5]+p for p in pay_with_coins(n-5) if (10 not in p) and (25 not in p) ] \
		 + [[1]+p for p in pay_with_coins(n-1) if (5 not in p) and (10 not in p) and (25 not in p)]
	else:
		return  [[1]+p for p in pay_with_coins(n-1) if (5 not in p) and (10 not in p) and (25 not in p)]
						
def pay_with_coins(cents):
	return pay_with_quarters(cents)
		
	
def pay_with_coins_of(cents,coin):
	n = 0
	ans = []
	while coin * n <= cents:
		ans += [ [n]+pay for pay in pay_with_dimes(cents-coin*n) ]
	return ans

def pay_with_quarters(cents):
	n = 0
	ans = []
	while 25 * n <= cents:
		ans += [ [n]+pay for pay in pay_with_dimes(cents-25*n) ]
		n += 1
	return ans

def pay_with_dimes(cents):
	n = 0
	ans = []
	while 10 * n <= cents:
		ans += [ [n]+pay for pay in pay_with_nickles(cents-10*n) ]
		n += 1
	return ans
	
def pay_with_nickles(cents):
	n = 0
	ans = []
	while 5 * n <= cents:
		ans += [ [n]+pay for pay in pay_with_pennies(cents-5*n) ]
		n += 1
	return ans

def pay_with_pennies(cents):
	return [[cents]]

	
def is_valid_chess_state(state):
	for i in range(len(state)):
		for j in range(i+1,len(state)):
			if state[i] == state[j]:
#				print 'same value: state[%d]=%d, state[%d]=%d' % (i, state[i], j, state[j])
				return False
			if abs(i-j) == abs(state[i]-state[j]):
#				print 'diag: state[%d]=%d, state[%d]=%d' % (i, state[i], j, state[j])
				return False
				
	return True
	
def make_moves(state,size):
	if not is_valid_chess_state(state):
		return []
		
	if  len(state) == size:
		return [[]]
	else:
		moves = []
		for i in range(1,size+1):
			moves += [[i]+s for s in make_moves(state+[i],size)]
		return moves	

def chess_queens(size):
	return make_moves([],size)

def is_smaller(b1,b2):
	return b1[0]<b2[0] and b1[1] < b2[1] and b1[2] < b2[2]
	
def stack_boxes(above, boxes, height_cache):
#	print above,boxes
	if above in height_cache:
		return height_cache[above]
		
	if boxes == []:
		return 0	
	else:
		heights = []
		for i in range(len(boxes)):
			if is_smaller(above,boxes[i]):
				rest = boxes[:]
				rest.pop(i)
				new_height = boxes[i][1] + stack_boxes(boxes[i], rest, height_cache)
				heights.append(new_height)
				
		if heights == []:
			return 0
		else:
			height_cache[above] = max(heights)
			return max(heights)
			

def make_boxes(n):
	return [(random.randrange(900),random.randrange(900),random.randrange(900)) for i in range(n)]
	
def longest_increasing(ls):
	for i in range(1,len(ls)):
		if (ls[i-1] > ls[i]):
			return i
	return None

def longest_decreasing(ls):
	for i in range(len(ls)-2,0,-1):
		if (ls[i] > ls[i+1]):
			return i
	return None

def find_min_max(ls,start,end):
	minimum = ls[start]
	maximum = ls[start]
	for i in range(start+1,end+1):
		if ls[i] < minimum:
			minimum = ls[i]
		if ls[i] > maximum:
			maximum = ls[i]
	return minimum, maximum
			
def shrink_to_left(ls,start,minimum):
	for i in range(start,1,-1):
		if ls[i-1] <= minimum:
			return i

def shrink_to_right(ls,start,maximum):
	for i in range(start,len(ls)-2):
		if ls[i+1] >= maximum:
			return i
	
def find_smallest_sort_block(ls):
	m = longest_increasing(ls)
	if m is None:
		return None, None
	n = longest_decreasing(ls)
	
	prev_min, prev_max = None, None
	minimum, maximum = find_min_max(ls,m,n)
	
	while (minimum != prev_min) or (maximum != prev_max):
		m = shrink_to_left(ls,m,minimum)
		n = shrink_to_right(ls,n,maximum)
		
		prev_min, prev_max = minimum, maximum
		minimum, maximum = find_min_max(ls,m,n)
	
	return m,n
	
def read_upto999(n):
	upto19 =['','one','two','three','four','five','six','seven','eight','nine', 'ten', \
	'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
	tens = ['','ten','tweenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
	
		
	q1 = n / 10
	d1 = n % 10
	
	q2 = q1 / 10
	d2 = q1 % 10
		
	q3 = q2 / 10
	d3 = q2 % 10
	
	if d3 > 0:
		hundred = ' hundred'
	else:
		hundred = ''
	
	if d2 < 2:
		num = '%s%s %s' % (upto19[d3], hundred, upto19[d2*10 + d1])
		#print d2, d1, d2*10+d1, num, upto19
	else:
		num = '%s%s %s %s' % (upto19[d3], hundred, tens[d2], upto19[d1])
	return  num.strip()
	
def read_num_in_english(n):
	k = [' trillion ',' billion ',' million ',' thousand ']
	
	num = read_upto999(n % 1000)
	while n / 1000 > 0:
		num = k.pop() + num
		n = n / 1000
		num = read_upto999(n % 1000) + num
		
	return num

def sequence_sum(ls, start, end):
	s = 0
	for i in range(start, end+1):
		s += ls[i]
	return s
	
def largest_sum_sequence(ls):
	maximum = - float('inf')
	i = 0
	j = 0
	while i < len(ls) and j < len(ls):
		val = sequence_sum(ls,i,j)
		#print i,j, val
		if val > maximum:
			maximum = val
			max_range = (i,j)
		if val < 0:
			i = j + 1
		j += 1
	return max_range
		
def count2s(n):
	cnt = 0
	if n % 10 == 2:
		cnt += 1
	while n / 10 > 0:
		n = n / 10
		if n % 10 == 2:
			cnt += 1
	return cnt
			
def count2s_upto(n):
	cnt = 0
	for i in range(n+1):
		cnt += count2s(i)
	return cnt
		
def cnt2s_on_ones(n):
	if n % 10  < 2:
		return n / 10
	else: 
		return n / 10 + 1

def cnt2s_on_tens(n):
	if n % 100 < 20:
		return 10 * (n / 100)
	elif n % 100 < 30:
		return 10 * (n / 100) + n % 10 + 1	
	else:
		return 10 * (n / 100) + 10

def cnt2s_on_hundreds(n):
	if n % 1000 < 200:
		return 100 * (n / 1000)
	elif n % 100 < 300:
		return 100 * (n / 1000) + n % 100 + 1	
	else:
		return 100 * (n / 1000) + 100
	
def cnt2s_on_pos(n,pos):
	'''
	pos = 1 for ones, 10 for tens, 100 for hundreds, ...
	'''
	if n % (10*pos) < 2*pos:
		return pos * (n / (10*pos))
	elif n % (10*pos) < 3*pos:
		return pos * (n / (10*pos)) + n % pos + 1	
	else:
		return pos * (n / (10*pos)) + pos
	
def cnt2s_upto(n):
	cnt = 0
	pos = 1
	while n / pos > 0:
		cnt += cnt2s_on_pos(n,pos)
		pos = pos * 10
		
	return cnt
	
INF = 999999999
def min_dist(ls, w1, w2):
	mdist = INF
	last_widx = None
	last_word = None
	for i, w in enumerate(ls):
		if w == w1:
			if last_word == w2:
				mdist = min(mdist, i-last_widx)
			last_word = w1
			last_widx = i
		elif w == w2:
			if last_word == w1:
				mdist = min(mdist, i-last_widx)
			last_word = w2
			last_widx = i

	return mdist
	
def longest_seq(seqs,maxv):
	L = 0
	seq = []
	
	for s in seqs:
		if s == [] or s[-1] > maxv:
			continue
		if len(s) > L:
			seq = s
			L = len(s)
			v = s[-1]
		elif len(s) == L:
			if s[-1] < v:
				seq = s
				v = s[-1]
	
	return seq[:]
	
def longest_inc_seq(x):
	if len(x) <= 1:
		return x
	
	lis = {}
	for maxv in x:
		if x[0] <= maxv:
			lis[maxv] = [x[0]]
		else:
			lis[maxv] = []
	#print 'init', lis
	
	new_lis = {}
	for i in x[1:]:
		for maxv in x:
			if i <= maxv:
				new_lis[maxv] = longest_seq(lis.values(),maxv)
				if new_lis[maxv] == [] or i > new_lis[maxv][-1]:
					new_lis[maxv].append(i)
			else:
				new_lis[maxv] = longest_seq(lis.values(),maxv) #lis[maxv]
			
			#print i, maxv, new_lis
			lis = new_lis
			
	return longest_seq(lis.values(),99999)

def longest_inc_subseq_min(ls,minv):
	#print ls, minv
	if ls == []:
		return ls
	else:
		if ls[0] >= minv:
			a = ls[0:1] + longest_inc_subseq_min(ls[1:],ls[0])
			b = longest_inc_subseq_min(ls[1:],minv)
			#print 'a,b =', a, b
			if len(a) >= len(b):
				 return a
			else:
				return b
		else:
			return longest_inc_subseq_min(ls[1:],minv)
		
def get_bottom_idx(m,n):
	if m == [] or m[0] > n:
		return -1
		
	start = 0
	end = len(m)
	while start < end:
		mid = (start + end) / 2 
		#print start,end,mid
		if m[mid] < n:
			if mid == end-1 or n < m[mid+1]:
				return mid
			start = mid + 1
		elif m[mid] > n:
			if mid == start or n >= m[mid-1]:
				return mid-1
			end = mid
		else:
			while m[mid] == n:
				mid += 1
			return mid
	
	return -1
		
def longest_inc_subseq(ls):
	#return longest_inc_subseq_min(ls,-999999)
	M = []
	for n in ls:
		j = get_bottom_idx(M,n)
		if j == -1:
			if M != []:
				M[0] = n
			else:
				M.append(n)
		elif j == len(M)-1: # if last number, expand
			M.append(n)
		elif M[j+1] > n:	#if n is better
			M[j+1] = n
	return len(M)
			
def factorial(n):
	ans = 1
	for i in range(1,n+1):
		ans = ans * i
	return ans
	
def num_permutations(s):
	letters = {}
	for c in s:
		letters[c] = letters.get(c,0) + 1
	num = factorial(len(s))
	for c in letters:
		num = num / factorial(letters[c])
	return num
			
	
def permutations(s):
	permu_set = set([])
	if len(s) == 1:
		return set(list(s))
		
	for i in range(len(s)):
		#print 'smaller string', s[:i]+s[i+1:]		
		smaller_permutations = permutations(s[:i]+s[i+1:])
		#print smaller_permutations, [s[i]+p for p in smaller_permutations]
		permu_set = permu_set.union(set([s[i]+p for p in smaller_permutations]))
		#print s, i, s[i], permu_set
	return permu_set
	
def get_befores(c,s):
	s_rests = {}
	for i in range(len(s)):
		if s[i] < c:
			s_rests[s[i]] = s[:i] + s[i+1:]
	return s_rests
	
def rank_permutations(s):
	s = s.lower()
	rank = 1
	for i in range(len(s)):
		befores  = get_befores(s[i],s[i:])
		for c in befores:
			rank += num_permutations(befores[c])
	return rank			

def cal_min_white_balls(nc,nb,p):
	'''
	nc : number of containers
	nb : number of black balls
	p : white ball percentage required
	
	IDEA
	The final distribution will be some containers with only black balls and the other containers each with only one white ball
	We start with distributing white balls as much as possible.
	If there are empty containers, we put a white ball to each of them. 
	If the percentage is met, we are done.
	If not, empty a container and put a white ball to it.
	It is better than putting a white ball into a container with black ball in terms of white ball percentage increase
	'''
	nbc = min(nc, nb)	# nbc: number of black ball containers. They have only black balls
	while nbc > 1: # do until only one black ball container is left
		if 100*(nc - nbc) >= nc*p:
			return nc - nbc 
		nbc -= 1

	# put white balls into the only black ball container until the percentage is met
	x = 1
	while 100* ((nc - 1) + 1.0*x/(x + nb))	< nc * p:
		x += 1
	return nc - 1 + x

import math
def cal_min_white_balls2(nc,nb,p):			
	 if 100*(nc-1) >= nc*p: 
	 	nw = int(math.ceil(nc*p/100.0))
	 	return max(nw, nc - nb)
	 else:
	 	return nc-1 + int(math.ceil(nb*(nc*p/100.0 - nc + 1)/(nc-nc*p/100.0)))
	 	#return nb*(nc*p/100.0 - nc + 1)/(nc-nc*p/100.0)
	
if __name__ == '__main__':
#	num_input = int(raw_input())
#	for i in range(num_input):
#		line = raw_input()
#		nc_str, nb_str, p_str = line.strip().split()
#		#print nc, nb, p
#		print cal_min_white_balls(int(nc_str),int(nb_str),int(p_str))
	
	print cal_min_white_balls2(1,1,60)
	print cal_min_white_balls2(2,1,60)
	print cal_min_white_balls2(10,2,50)
	print cal_min_white_balls2(70,70,70)
	print cal_min_white_balls2(65,50,58)
	#print rank_permutations('QUESTION')
	#print permutations('abcab')
	
	#print longest_seq([[11,12],[3],[2,4]], 3)
	#print longest_inc_subseq([0, 8, 4])
	#print longest_inc_subseq([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15])
	#print longest_inc_seq([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15])
	#print get_bottom_idx([1, 3, 4, 7, 8, 8, 8, 10, 23], 8)
	#print longest_inc_subseq([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15])
	#print min_dist(['aa', 'ab', 'ac', 'bb', 'bc', 'bd', 'be', 'aa'], 'aa', 'bc')
	#print cnt2s_on_tens(1128), cnt2s_on_pos(1128,10)
	#print cnt2s_upto(122900000)
	#print count2s_upto(122900000)
	#print largest_sum_sequence([-10])
	#print largest_sum_sequence([2, -8, 3, -2, 4, -10])
	#print read_num_in_english(2849829835265)
	#print find_min_max([1,2,4,7,10,11,7,12,6,7,16,18,19],3,8)
	#print shrink_to_right([1,2,4,7,10,11,7,12,6,7,16,18,19],7,12)
	#print find_smallest_sort_block([1,2,4,7,10,11,3,17,6,7,16,18,19])
	
	#boxes = make_boxes(700)
	#print boxes, '\n', stack_boxes((0,0,0),boxes,{})
	#print chess_queens(8)
	#print is_valid_chess_state([5,2,4,1,3])
	#print is_valid_chess_state([5,2,3,1,3])
	#print pay_with_nickles(7)
	#print pay_with_coins(32)	
	#print get_parens(3)
	#print get_permutations('abcdefg')
	#print get_subsets2([1,2,3])
#	print num_shortest_paths_dp(500,500)
#	print num_shortest_paths(15,15)

#	for i in range(1,50):
#		print get_multiple_primes(i), exponents2num(get_multiple_primes(i))
		
#	print fibonacci_dp2(40)
	
# print stair_hops(10)
#	print stair_hops_dp(40)
#	print pick_random_from_stream()
	
#	t1 = Tree(1,Tree(2,Tree(3),Tree(4)), Tree(5,Tree(6),Tree(7)))
#	t2 = Tree(1,None, Tree(5,Tree(6),Tree(7)))
#	t3 = Tree(5,Tree(3,Tree(2),Tree(4)), Tree(8,Tree(6),Tree(9)))
#	print is_binary_search_tree(t3)
#	print get_nodes_per_level(t1)
#	print i s_balanced(t2)
#	add_right_neighbor(t1)
#	print t1.show()
#	ls = list('abcdefghimnopqrstuvwxyz')
#	print has_duplicate_char(ls)
#	ls = [3,2,5,4,1]
#	quicksort_inplace(ls)
#	print ls
#	print compress_string("aabcaa")