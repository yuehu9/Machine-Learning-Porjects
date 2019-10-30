from gridMDPmodel import gridModel, randomPlayer
import numpy as np

if __name__ == '__main__':
	# MDP for party problem
	states = ['RU8p', 'TU10p', 'RU10p', 'RD10p', 'RU8a', 'RD8a', 'TU10a', 'RU10a', 'RD10a', 'TD10a', '11am class begins']
	actions = ['P', 'R', 'S']

	rewards = [[2, 0, 1], [2, 0, np.nan], [2, 0, -1],
			   [2, 0, np.nan], [2, 0, -1], [2, 0, np.nan],
			   [-1, -1, -1], [0, 0, 0], [4, 4, 4], [3, 3, 3]]
	transition = [[([1],[1]), ([2],[1]), ([3],[1])], [([7],[1]), ([4],[1]), ()], [([4, 7],[0.5, 0.5]), ([4],[1]), ([5],[1])],
				  [([5, 8],[0.5, 0.5]), ([5],[1]), ()], [([6],[1]), ([7],[1]), ([8],[1])], [([9],[1]), ([8],[1]), ()],
				  [([10],[1]), ([10],[1]),([10],[1]),([10],[1])], [([10],[1]), ([10],[1]),([10],[1]),([10],[1])],
				  [([10],[1]), ([10],[1]),([10],[1]),([10],[1])], [([10],[1]), ([10],[1]),([10],[1]),([10],[1])]]
	startState = 0
	endState = 10

	#### Q1 ######
	print('Part 1: run 50 episodes and see experience sequence.')
	print(' (curent state, action) --> (rewards, next state)')
	MDP = gridModel(states, actions, rewards, transition, startState, endState)
	Rplayer = randomPlayer(MDP)
	Rplayer.runEpoches(50)

	print('\nState values for random policy: \n')
	allActions = [MDP.getLegalActions(s) for s in range(MDP.numState - 1)]
	randomPolicy = [(a, [1/len(a) for _ in range(len(a))]) for a in allActions]
	stateValues = MDP.policyEvaluation(randomPolicy)
	MDP.printValue(stateValues)

	#### Q2 #######
	print('\nPart 2: learning optimal policy by policy iteration')
	initP = [([0], [1]) for s in range(MDP.numState - 1)]
	MDP.policyIteration(initP)