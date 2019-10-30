import numpy as np

class gridModel:
	def __init__(self, states, actions, rewards, transition, startState, endState):
		"""
		A model fully describes the Markov transition and the rewards of the environment. 
		Use grid to do table lookup
		states: list of all distinct states, ordered in accordance with the matrix grid
		actions: list of all actions, ordered in accordance with the matrix grid
		rewards: state*acton matrix grid, real number entries. reward is nan if acrion is illegal
		transition: state*acton matrix grid. Entries are list of (stateInd, possiblity) pairs
		"""
		self.states = states
		self.actions = actions
		self.rewards = rewards
		self.transition = transition
		self.startState = startState
		self.endState = endState
		self.numAct = len(actions)
		self.numState = len(states) 

	def getLegalActions(self, state):
		act = [a for a in range(self.numAct) if not np.isnan(self.rewards[state][a])]
		return act

	def isEndState(self, state):
		return state == self.endState

	def getStartState(self):
		return self.startState

	def getNextState(self, curstate, act):
		listS, posS =  self.transition[curstate][act]
		if len(listS) == 1:
			return listS[0]
		else:
			return np.random.choice(listS, p = posS)

	def allNextStates(self, curstate, act):
		return self.transition[curstate][act]

	def getActionReward(self, curstate, act):
		return self.rewards[curstate][act]

	def policyEvaluation(self, policy, initV = [], thresh = 0.001):
	# policy: a list that recoreds policy for every state. entries are (list of actions, possiblilities) pair
		if initV == []:
			stateValues = np.zeros(self.numState)
		else:
			stateValues = initV
		delta = float('inf')
		while delta > thresh:
			delta = 0
			# donnot update value for terminal state
			for s in range(self.numState - 1):
				oldV = stateValues[s]
				actions, actionP = policy[s]
				v = 0
				for a, pA in zip(actions, actionP):
					vA = self.getActionReward(s, a)
					nextS, p = self.allNextStates(s, a)
					for ns, nP in zip(nextS, p):
						vA += stateValues[ns] * nP
					v += vA * pA
				stateValues[s] = v
				delta = max(delta, abs(stateValues[s] - oldV))
		return stateValues

	def policyIteration(self, initP):
		stateValues = np.zeros(self.numState)
		policy = initP
		stable = False
		k = 0
		while not stable:
			print('iteration: ', k)
			# policy evaluation
			stateValues = self.policyEvaluation(policy)
			self.printValue(stateValues)

			# policy Improvement
			stable = True
			for s in range(self.numState - 1):
				oldAction = policy[s]
				actions = self.getLegalActions(s)
				# find best action
				bestA = None
				bestV = float('-inf')
				for a in actions:
					vA = self.getActionReward(s, a)
					nextS, p = self.allNextStates(s, a)
					for ns, nP in zip(nextS, p):
						vA += stateValues[ns] * nP
					if vA > bestV:
						bestV = vA
						bestA = a
				policy[s] = ([bestA], [1])
				# check convergence
				if oldAction != policy[s]:
					stable = False
			self.printDetermPolicy(policy)
			k += 1
		return policy

	def printDetermPolicy(self, policy):
		# print deterministic policy
		str = 'Policy: '
		for s in range(self.numState - 1):
			a, _ = policy[s]
			str +=' (' + self.states[s] + ': ' + self.actions[a[0]] + ') '
		print(str)

	def printValue(self, stateValues):
		str = 'State values: '
		for i in range(self.numState):
			str += ' ({} : {:0.2f}) '.format(self.states[i], stateValues[i])
		print(str)


class randomPlayer:
	# choose legal actions uniform random
	def __init__(self, gridModel):
		self.model = gridModel # the player knows the model

	def selectAction(self, state):
		# decide next action by randomly choosing legal actions based on curent state
		act = self.model.getLegalActions(state)
		return np.random.choice(act)

	# output the history log and final return for one epoch
	def runOneEpoch(self):
		curState = self.model.getStartState()
		totalReturn = 0
		history = ''
		while not self.model.isEndState(curState):
			act = self.selectAction(curState)
			r = self.model.getActionReward(curState, act)
			nextS = self.model.getNextState(curState, act)
			curH = ' (' + self.model.states[curState] + ' ' + self.model.actions[act]+ '-->' + str(r) + ' ' + self.model.states[nextS] + ') '
			history += curH
			totalReturn += r
			curState = nextS
		return totalReturn, history

	# output results of n numbers of epoches
	def runEpoches(self, n):
		avgReturn = 0
		for _ in range(n):
			epReturn, epHistory = self.runOneEpoch()
			avgReturn += epReturn
			print(epHistory)
		avgReturn /= float(n)
		print("Average return for {} epoches is {:0.3f}".format(n, avgReturn))














