import numpy as np

class dicitonaryModel:
	def __init__(self, states, rewards, transition, legalActions, startState, endState):
		"""
		A model fully describes the Markov transition and the rewards of the environment. 
		Use dicitonary to do value lookups.
		states: list of all distinct states
		rewards: a dectionary of (state, action) pair to a reward.
		transition: a dicironary of (state, action) pair to (state, possibility) pair
		legalActions: a dicitonary of State to list of legal actions
		"""
		self.states = states
		self.rewards = rewards
		self.transition = transition
		self.legalActions = legalActions
		self.startState = startState
		self.endState = endState

	def getLegalActions(self, state):
		return self.legalActions[state]

	def isEndState(self, state):
		return state == self.endState

	def getStartState(self):
		return self.startState

	def getNextState(self, curstate, action):
		return self.transition((curstate, action))

	def getActionReward(self, curstate, action):
		return self.rewards((curstate, action))


class randomPlayer:
	def __init__(self, dictModel):
		self.model = dictModel # the player knows the model

	def selectAction(self, state):
		# decide next action by randomly choosing legal actions based on curent state
		act = self.model.getLegalActions(state)
		ind = np.random.randint(0, len(act))  
		return act[int]

	def runOneEpoch(self):
		curState = self.model.getStartState()
		totalReturn = 0
		history = ''
		while not self.model.isEndState(curState):
			act = self.selectAction(curState)
			r = self.model.getActionReward(curState, act)
			nextS = self.model.getNextState(curState, act)
			curH = ' (' + curState + ' ' + act + '-->' + str(r) + ' ' + nextS + ') '
			history += curH
			totalReturn += r
			curState = nextS
		return totalReturn, history

	def runEpoches(self, n):
		avgReturn = 0
		for _ in range(n):
			epReturn, epHistory = self.runOneEpoch()
			avgReturn += epReturn
			print(epHistory)
		avgReturn /= float(n)
		print("average return for {} epoches is {}".format(n, avgReturn))














