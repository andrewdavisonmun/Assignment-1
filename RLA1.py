#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 14:43:14 2025

@author: andrewdavison
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


k = 10 # 10 bandits (k)
n_steps = 2000 # 2000 steps
n_simulations = 1000 # 1000 simulations

'''The bandit fuction has settings to produce the greedy bandit, epsilon-greedy bandit
and the optimistic bandit. 
sims - the number of simulations
optimistic - boolean if we are doing optimistic start or not
epsilon - epsilon for e-greedy. if not set it will not be considered
means_input - this allows for the generated means for the experiments in part 2,
will be None for part 1. We want for each seperate test to have the same means through the
test so we will pass in a means array for part 2
reset - this allows the reset in part 2'''
def bandit(sims,optimistic, epsilon = -1, means_input = None, reset = False):
    # initialize rewards and best_chosen as 0 over the number of steps
    rewards = np.zeros(n_steps)
    best_chosen = np.zeros(n_steps)
    
    base_seed = 42
    # we run it for every run in n_simulations
    for run in range(sims):
        rng = np.random.default_rng(base_seed + run) # creates a different random stream for each run. each 
        
        
        if means_input is None: # is None for our part 1 tests
            means = np.tile(rng.normal(0, 1, k),(n_steps,1)) # creates the means for part 1
        else:
            means = means_input # otherwise use means from input
        if optimistic:
            # if optimistic use the normal distribution 99.5 percentile with the max mean on the first step
            optval = norm.ppf(0.995, loc = np.max(means[0]) ,scale =1)
            # Q is the estimated action values per arm. The sample average of past rewards per arm
            Q = np.ones(k) * optval # we set that view for each to a very optimistic one in optimistic start
        else:
            Q = np.zeros(k) # set to 0 in normal start for each bandit
        N = np.zeros(k) # N is the number of times that we have chosen each bandit. starts at 0
        
        # we run it for every step in n_steps
        for step in range(n_steps):
            if reset == True: # add the reset feature
                if step == 501: # at step 501 we reset the our previous view
                    if optimistic: # same as before
                        optval = norm.ppf(0.995, loc = np.max(means[0]) ,scale =1)
                        Q = np.ones(k) * optval
                    else:
                        Q = np.zeros(k)
                    N = np.zeros(k) # as we haven't chosen anything with the new view set each to 0
            # we track the if the best action is taken in each step.
            # the best action is the best mean that we could have taken from rather than what would've returned the best reward
            best_action = np.flatnonzero(means[step] == np.max(means[step]))
            
            # this is the epsilon-greedy implementation if a random number is less than epsilon then select randomly
            if epsilon >=0 and rng.random() < epsilon: #if epsilon is below 0 then it is just greedy (default)
                action = rng.integers(k)
            else:
                action = rng.choice(np.flatnonzero(Q == Q.max())) # else pick the max Q (random tiebreaker)
            reward = rng.normal(means[step, action], 1) # select the reward randomly from the mean at bandit (action) location
            
            N[action] += 1 # increment N by 1
            Q[action] += (reward - Q[action]) / N[action] # Use sample average to update action-value estimates
            
            rewards[step] += reward # we track the reward for each step (added together from all the simulations)
            # if we chose one of the best actions then add 1 to the best_chosen for that step number
            if action in best_action:
                best_chosen[step] += 1
    return rewards/sims, best_chosen/sims # return our tracked information averaged by number of simulations

"""Gradient bandit algorithm
sims - the number of simulations
alpha - the learning rate. Changes how fast the algorithm adapts to new rewards. Higher alpha adapts faster to recent rewards
baseline - used for tuning and exploration of how changing the baseline effects the algorith.
Moving Baseline is the expected reward or average of observed rewards (all rewards seen/n actions total). default
No Baseline means we aren't using the new rewards at all. for exploration
Fixed Baseline means the baseline does not change so we look at rewards but don't comapre with a baseline
means_input - this allows for the generated means for the experiments in part 2,
will be None for part 1. We want for each seperate test to have the same means through the
test so we will pass in a means array for part 2
reset - this allows the reset in part 2"""
def gradient_bandit(sims, alpha, baseline = 'moving', means_input = None, reset = False):
    # initialize rewards and best_chosen as 0 over the number of steps
    rewards = np.zeros(n_steps)
    best_chosen = np.zeros(n_steps)
    
    base_seed = 42
    # we run it for every run in n_simulations
    for run in range(sims):
        
        rng = np.random.default_rng(base_seed + run) # creates a different random stream for each run
        if means_input is None: # is None for our part 1 tests
            means = np.tile(rng.normal(0, 1, k),(n_steps,1)) # creates the means for part 1
        else:
            means = means_input # otherwise use means from input
        
        # H is our preference of each bandit. We start at 0 for each bandit
        H = np.zeros(k)
        avg_reward = 0 # initialize avg_reward
        
        # we run it for every step in n_steps
        for step in range(n_steps):
            if reset == True: # add the reset feature
                if step == 501: # at step 501 we reset the our previous view
                    H = np.zeros(k) # reset preference policy to zero
                    avg_reward = 0 # average reward back to 0
            
            # we track the if the best action is taken in each step.
            # the best action is the best mean that we could have taken from rather than what would've returned the best reward
            best_action = np.flatnonzero(means[step] == np.max(means[step]))
            # we need the H as an exponent (base e) in the policy formula.
            # We can subtract by max H for numerical stability without affecting the outcome
            exp_H = np.exp(H - np.max(H))
            pi = exp_H / np.sum(exp_H) # calculate the policy using the formula
            action = rng.choice(k, p = pi) # choose the action based on the policy
            
            reward = rng.normal(means[step, action], 1) # calculate the reward based on normal distribution
            #this is the baseline selection implementation
            if baseline == 'moving':
                avg_reward += (reward - avg_reward) / (step+1) #moving changes the reward based on each of the previous
                baseline_val = avg_reward
            elif baseline == 'fixed':
                baseline_val = 0 # fixed keeps the baseline at 0
            elif baseline == 'none':
                baseline_val = None # a none baseline does not factor in reward and is not really a gradient bandit. Experiment
            
            # our no baseline is mostly for experimental purposes
            if baseline_val is None:
                for a in range(k):
                    if a == action:
                        H[a] += alpha * (1 - pi[a]) # 'No baseline' doesn't use reward calculation . only based on probabilities
                    else:
                        H[a] -= alpha * pi[a] # update for non chosen H
            else:
                for a in range(k):
                    if a == action:
                        H[a] += alpha * (reward - baseline_val) * (1 - pi[a]) # update H based on update formula 
                    else:
                        H[a] -= alpha * (reward - baseline_val) * pi[a] # update H for non chosen bandits
            
            rewards[step] += reward # we track the reward for each step (added together from all the simulations)
            # if we chose one of the best actions then add 1 to the best_chosen for that step number
            if action in best_action:
                best_chosen[step] += 1
                
    return rewards/sims, best_chosen/sims # return our tracked information averaged by number of simulations

"""print function that prints a summary of the results of the simulation
label - the name of the experiment
rewards - the average reward for each step
top_actions - percentage (in decimal) of times each top action was chosen for each step

The reward is an average of the last 100 steps in the simulation to reduce bias by increasing the amount of samples
"""
def print_summary(label, rewards, top_actions):
    avgreward = np.mean(rewards[-100:])
    avgoptact = np.mean(top_actions[-100:]) * 100
    print(f"{label}: Final Reward avg: {avgreward:.4f}, Final Best Action % = {avgoptact:.2f}%")

"""plot the results
results - the reward average over steps and best action percentage (decimal) over steps
title - the title of the graph

we are reporting the action percentage so we multiply by 100"""
def plot_results(results, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, (rewards, action) in results.items():
        plt.plot(rewards,label = label)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Steps')
    plt.legend(loc='best')


    plt.subplot(1,2,2)
    for label, (rewards, action) in results.items():
        plt.plot(action*100,label = label)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title('% Optimal Action over Steps')
    plt.legend(loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) #add room for title
    plt.suptitle(title, fontsize = 18) #add title
    plt.show()


results_eg = {}
epsilons = [0.01, 0.05, 0.1, 0.2, 0.3] # epsilons to try

print("Epsilon-Greedy tuning")
for eps in epsilons:
    label = f"Epsilon={eps}"
    results_eg[label] = bandit(650,False,eps) # run bandit for each epsilon
    rewards, top_actions = results_eg[label]
    print_summary(label, rewards, top_actions) # call print summary

plot_results(results_eg, "Epsilon-Greedy Tuning") # plot the results of the tuning

results_grad = {}
baselines = ['moving','fixed','none'] # baselines to try
alphas = [0.01, 0.05, 0.1, 0.2] # alphas to try

print("Gradient tuning")
for alphs in alphas:
    for bases in baselines:
        label = f"Alpha={alphs}/baseline = {bases}"
        results_grad[label] = gradient_bandit(650,alphs,bases)
        rewards, top_actions = results_grad[label]
        print_summary(label, rewards, top_actions)

plot_results(results_grad, "Gradient Tuning") # plot results

results = {}

print("Greedy")
results['Greedy'] = bandit(n_simulations,False)

print("Epsilon-Greedy with 0.05")
results['Epsilon'] = bandit(n_simulations,False,0.05) # 0.05 found to be best

print("Optimistic Starting Values")
results['Optimistic'] = bandit(n_simulations,True)

print("Gradient with 0.05")
results['Gradient'] = gradient_bandit(n_simulations,0.05,'moving') # 0.05 and moving best

for label, (rewards, top_actions) in results.items():
    print_summary(label, rewards, top_actions)

plot_results(results, "Final Results")

plt.show()

"""apply the drift to the means. return means with drift
seed - the random seeds for each arm"""
def apply_drift(seed):
    means = np.zeros((n_steps,k)) # set the means to 0 for all
    for arm in range(k): # for each arm
        rng = np.random.default_rng(seed[arm]) # get rng for arm
        means[0,arm] = rng.normal(0,1) # get the first mean for the arm
        for t in range(1, n_steps):
            means[t,arm] = means[t-1,arm] + rng.normal(0,0.01) # change the mean slightly based on N(0,0.01^2)
    return means

"""apply the drift and mean revert to the means. return means with drift
seed - the random seeds for each arm"""
def apply_mean_revert(seed):
    means = np.zeros((n_steps,k)) # set the means to 0 for all
    for arm in range(k): # for each arm
        rng = np.random.default_rng(seed[arm]) # get rng for arm
        means[0,arm] = rng.normal(0,1) # get the first mean for the arm
        for t in range(1, n_steps):
            means[t,arm] = 0.5 * means[t-1,arm] + rng.normal(0,0.01) # change the mean by applying the mean revert. at K = 0.5
    return means

"""apply the drift and mean revert to the means. return means with drift
seed - the random seed
perm - the set permutation of the swap"""
def apply_abrupt(seed, perm):
    rng = np.random.default_rng(seed) # set the seed
    means = np.zeros((n_steps,k)) # set the means to 0
    orig_means = rng.normal(0,1,k) # randomize the means
    for i in range(501):
        means[i] = orig_means # means for first 501 are the calculated
    perm_means = orig_means[perm] # permutate the original means
    for i in range(501, n_steps):
        means[i] = perm_means # means after 501 are permuted version of original means. All means after the 501th are permuted
    return means

np.random.seed(42)
drift_seeds = np.random.randint(0,100000,size = 10) # set seeds for each arm
rng = np.random.default_rng(42) # set rng for permutating
perm = rng.permutation(k) # set the permuation

driftmean = apply_drift(drift_seeds) # get the means for the drifted bandits
revertmean = apply_mean_revert(drift_seeds) # get the means for the mean revert bandits
abruptmean = apply_abrupt(42,perm) # get the means for the abrupt changes

# do similar to above. tuning and producing results from experiment of the drift mean
# only using alphas and epsilons of 0.05 and 0.1 as those were the best performing by a fair margin
print("Drift")

results_eg = {}
epsilons = [0.05, 0.1]

print("Epsilon-Greedy tuning")
for eps in epsilons:
    label = f"Epsilon={eps}"
    results_eg[label] = bandit(325,False,eps, means_input=driftmean)
    rewards, top_actions = results_eg[label]
    print_summary(label, rewards, top_actions)

plot_results(results_eg, "Epsilon-Greedy Tuning Drift")

results_grad = {}
baselines = ['moving','fixed','none']
alphas = [0.05, 0.1]

print("Gradient tuning")
for alphs in alphas:
    for bases in baselines:
        label = f"Alpha={alphs}/baseline = {bases}"
        results_grad[label] = gradient_bandit(325,alphs,bases, means_input=driftmean)
        rewards, top_actions = results_grad[label]
        print_summary(label, rewards, top_actions)

plot_results(results_grad, "Gradient Tuning Drift")

results = {}

print("Greedy")
results['Greedy'] = bandit(n_simulations,False, means_input=driftmean)

print("Epsilon-Greedy with 0.05")
results['Epsilon'] = bandit(n_simulations,False,0.05, means_input=driftmean)

print("Optimistic Starting Values")
results['Optimistic'] = bandit(n_simulations,True, means_input=driftmean)

print("Gradient with 0.05")
results['Gradient'] = gradient_bandit(n_simulations,0.05,'moving', means_input=driftmean)


for label, (rewards, top_actions) in results.items():
    print_summary(label, rewards, top_actions)

plot_results(results, "Drift")

plt.show()

# do similar to above. tuning and producing results from experiment of the mean revert
# only using alphas and epsilons of 0.05 and 0.1 as those were the best performing by a fair margin
print("Mean Revert")

results_eg = {}
epsilons = [0.05, 0.1]

print("Epsilon-Greedy tuning")
for eps in epsilons:
    label = f"Epsilon={eps}"
    results_eg[label] = bandit(325,False,eps, means_input=revertmean)
    rewards, top_actions = results_eg[label]
    print_summary(label, rewards, top_actions)

plot_results(results_eg, "Epsilon-Greedy Tuning Mean Revert")

results_grad = {}
baselines = ['moving','fixed','none']
alphas = [0.05, 0.1]

print("Gradient tuning")
for alphs in alphas:
    for bases in baselines:
        label = f"Alpha={alphs}/baseline = {bases}"
        results_grad[label] = gradient_bandit(325,alphs,bases, means_input=revertmean)
        rewards, top_actions = results_grad[label]
        print_summary(label, rewards, top_actions)

plot_results(results_grad, "Gradient Tuning Mean Revert")

results = {}

print("Greedy")
results['Greedy'] = bandit(n_simulations,False, means_input=revertmean)

print("Epsilon-Greedy with 0.05")
results['Epsilon'] = bandit(n_simulations,False,0.05, means_input=revertmean)

print("Optimistic Starting Values")
results['Optimistic'] = bandit(n_simulations,True, means_input=revertmean)

print("Gradient with 0.05")
results['Gradient'] = gradient_bandit(n_simulations,0.05,'moving', means_input=revertmean)


for label, (rewards, top_actions) in results.items():
    print_summary(label, rewards, top_actions)

plot_results(results, "Mean Revert")

plt.show()

# do similar to above. tuning and producing results from experiment of the abrupt change
# only using alphas and epsilons of 0.05 and 0.1 as those were the best performing by a fair margin
print("Abrupt Change")

results_eg = {}
epsilons = [0.05, 0.1]

print("Epsilon-Greedy tuning")
for eps in epsilons:
    label = f"Epsilon={eps}"
    results_eg[label] = bandit(325,False,eps, means_input=abruptmean)
    rewards, top_actions = results_eg[label]
    print_summary(label, rewards, top_actions)

plot_results(results_eg, "Epsilon-Greedy Tuning Abrupt Change")

results_grad = {}
baselines = ['moving','fixed','none']
alphas = [0.05, 0.1]

print("Gradient tuning")
for alphs in alphas:
    for bases in baselines:
        label = f"Alpha={alphs}/baseline = {bases}"
        results_grad[label] = gradient_bandit(325,alphs,bases, means_input=abruptmean)
        rewards, top_actions = results_grad[label]
        print_summary(label, rewards, top_actions)

plot_results(results_grad, "Gradient Tuning Abrupt Change")

results = {}

print("Greedy")
results['Greedy'] = bandit(n_simulations,False, means_input=abruptmean)

print("Epsilon-Greedy with 0.05")
results['Epsilon'] = bandit(n_simulations,False,0.05, means_input=abruptmean)

print("Optimistic Starting Values")
results['Optimistic'] = bandit(n_simulations,True, means_input=abruptmean)

print("Gradient with 0.05")
results['Gradient'] = gradient_bandit(n_simulations,0.05,'moving', means_input=abruptmean)


for label, (rewards, top_actions) in results.items():
    print_summary(label, rewards, top_actions)

plot_results(results, "Abrupt Change")

plt.show()

# do similar to above. tuning and producing results from experiment of the abrupt change with reset
# only using alphas and epsilons of 0.05 and 0.1 as those were the best performing by a fair margin
print("Abrupt Change with Reset")

results_eg = {}
epsilons = [0.05, 0.1]

print("Epsilon-Greedy tuning")
for eps in epsilons:
    label = f"Epsilon={eps}"
    results_eg[label] = bandit(325,False,eps, means_input=abruptmean,)
    rewards, top_actions = results_eg[label]
    print_summary(label, rewards, top_actions)

plot_results(results_eg, "Epsilon-Greedy Tuning Abrupt Change Reset")

results_grad = {}
baselines = ['moving','fixed','none']
alphas = [0.05, 0.1]

print("Gradient tuning")
for alphs in alphas:
    for bases in baselines:
        label = f"Alpha={alphs}/baseline = {bases}"
        results_grad[label] = gradient_bandit(325,alphs,bases, means_input=abruptmean,)
        rewards, top_actions = results_grad[label]
        print_summary(label, rewards, top_actions)

plot_results(results_grad, "Gradient Tuning Abrupt Change Reset")

results = {}

print("Greedy")
results['Greedy'] = bandit(n_simulations,False, means_input=abruptmean, reset = True)

print("Epsilon-Greedy with 0.05")
results['Epsilon'] = bandit(n_simulations,False,0.05, means_input=abruptmean, reset = True)

print("Optimistic Starting Values")
results['Optimistic'] = bandit(n_simulations,True, means_input=abruptmean, reset = True)

print("Gradient with 0.05")
results['Gradient'] = gradient_bandit(n_simulations,0.05,'moving', means_input=abruptmean, reset = True)


for label, (rewards, top_actions) in results.items():
    print_summary(label, rewards, top_actions)

plot_results(results, "Abrupt Change with Reset")

plt.show()