from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
#[ 0.8216812   2.40384327  0.06281146 10.63500319]

class AngledCustomController(FlightController):
    def __init__(self):
        """
        Add params for the controller:
        """
        self.parameters = [0,0,0,0]
        self.C = self.parameters[0]  # constant
        self.X = self.parameters[1]  # drag coeff
        self.Y = self.parameters[2]
        self.S = self.parameters[3]
        self.pop_size = 4**4 # num of solutions
        self.number_generations = 1

    def run_simulation(self, parameters):
        
        drone = self.init_drone()
        target = drone.get_next_target()  
        rewards = 0
        r_min = np.sqrt((drone.x-target[0])**2+(drone.y-target[1])**2) 
        count = 0

        for t in range(self.get_max_simulation_steps()): 
            
            drone.set_thrust(np.clip(self.get_thrusts(drone, parameters), 0, 1))
            target_hit = drone.step_simulation(self.get_time_interval())
            r = np.sqrt((drone.x-target[0])**2+(drone.y-target[1])**2)

            if target_hit:
                rewards+=800
                target = drone.get_next_target()  
                r_min = np.sqrt((drone.x-target[0])**2+(drone.y-target[1])**2) 
                count +=1
                print("HIT")
            elif r<(r_min-0.1):
                rewards+=1/r # this line is the old version
                r_min = r
            else:
                rewards -=1/r  
        if (count >= 2):
            print(count, " targets hit!")#
            print(parameters)
        return rewards, count
    
    def crossover(self, survivors):
        # crossover random pairs of survivors
        crossover_solutions = np.empty((0,4))
        solution_inds = range(len(survivors[:,0]))
        for i in solution_inds:
            random_pair = np.random.choice(solution_inds, 2, replace = False)
            sol_1 = survivors[random_pair[0],:]
            sol_2 = survivors[random_pair[1],:]  
            parent_sols = np.vstack((sol_1, sol_2))
            x_over_sols = np.empty((2,4))
            for g in range(4):
                x_over_sols[:,g] = np.random.choice(parent_sols[:,g], 2)

            crossover_solutions = np.vstack((crossover_solutions, x_over_sols))
        return crossover_solutions
    
    def mutate(self, survivors, generation):
        mutated_solutions = np.empty((0,4))
        
        mutation_rates = [0.3,5,0.3,5]
        for i in range(len(survivors[:,0])):
            mutations = np.empty((5,4))
            for g in range(4):
                sd = mutation_rates[g]*(1.5-generation/self.number_generations) 
                mutations[:,g] = abs(np.random.normal(survivors[i,g],sd, 5)) #   FIX VARIANCE
            mutated_solutions[:,0] = np.clip(mutated_solutions[:,0], 0, 1)
            mutated_solutions[:,2] = np.clip(mutated_solutions[:,2], 0, 1)

            mutated_solutions = np.vstack((mutated_solutions, mutations))
        return mutated_solutions
    
    def breed_survivors(self,survivors, generation):
        crossover_solutions = self.crossover(survivors)
        mutated_solutions = self.mutate(survivors, generation)     
        new_solutions = np.vstack((crossover_solutions, mutated_solutions))
        solutions = np.vstack((survivors, new_solutions))
        return solutions
    
    def select_survivors(self, solution_rewards, solutions):
        probabilities = np.array(solution_rewards - min(solution_rewards)).astype(float)       
        solution_inds = range(len(solutions[:,0]))
        if sum(probabilities) == 0:
            probabilities[:] = 1/self.pop_size
        else:
            probabilities = probabilities**2/sum(probabilities**2)
        # select 10% of population to create next generation

        survivor_inds = np.random.choice(solution_inds, int(self.pop_size/8),
                                         p = probabilities)
        survivors = np.clip(solutions[survivor_inds], 0, None)
        return survivors

    def train(self, number_of_episodes):
        #start by setting random parameters, for 100 solutions    
        self.number_generations = number_of_episodes
        C_vals = np.linspace(0, 1, int(self.pop_size**0.25)) # hovering constant
        X_vals = Y_vals = np.linspace(0, 100, int(self.pop_size**0.25)) # direction constants
        S_vals = np.linspace(0,10, int(self.pop_size**0.25)) # rotation stabalisation constants
        Y_vals = C_vals
        CC, XX, YY, SS = np.meshgrid(C_vals, X_vals, Y_vals, S_vals)
        solutions = np.vstack((CC.flatten(), XX.flatten(), 
                               YY.flatten(), SS.flatten()))
        solutions = solutions.transpose() 
        survivors = []
        # double up starting values to get kx, bx, k_theta, b_theta     
        average_rewards = []
        gen_closest_approaches = []
        for generation in range(self.number_generations):
            print(generation)
            simulation_analysis = np.apply_along_axis(self.run_simulation, axis = 1, arr = solutions) 
            #sim analysis is a list of tuples (rewards, r_min)
            solution_rewards, targets_hit = simulation_analysis[:,0], simulation_analysis[:,1]
            survivors = self.select_survivors(solution_rewards, solutions)
            solutions = self.breed_survivors(survivors, generation)
            
            average_rewards.append(np.mean(solution_rewards))
            gen_closest_approaches.append(np.mean(targets_hit))
        
        return solutions, average_rewards, gen_closest_approaches, survivors
            
    def get_thrusts(self, drone: Drone, parameters) -> Tuple[float, float]:
        
        C,X,Y,S = parameters        
        target = drone.get_next_target()
        dx = (target[0] - drone.x)
        dy = (target[1] - drone.y)
    
        try:
            theta_target = np.arctan(dx/abs(dy))
        except ZeroDivisionError:
            theta_target = 0
        theta_rel = theta_target - drone.pitch

        if dy>0:
            T1 = C + X*theta_rel - S*drone.pitch_velocity
            T2 = C - X*theta_rel + S*drone.pitch_velocity
        else:
            T1 = C/2 + X*theta_rel - S*drone.pitch_velocity
            T2 = C/2 - X*theta_rel + S*drone.pitch_velocity
        return np.clip([T1,T2], 0, 1)
            
    def load(self):
        pass
    def save(self):
        pass
    
def run_for(episodes):
    c = AngledCustomController()
    sols, gen_rs, rs, survivors = c.train(episodes)
    return gen_rs, sols, rs, survivors



if __name__ == "__main__":
    episodes = 10
    gens, sols, rs, survivors = run_for(episodes)
    
    sol = np.mean(survivors, axis = 0)
    
    
    fig = plt.figure()
    gs = fig.add_gridspec(2,2,)
    rewards = fig.add_subplot(gs[0,:])
    rewards.plot(gens, color = "purple")
    rsp = rewards.twinx()
    rsp.plot(rs, color = "cyan")
    p01 = fig.add_subplot(gs[1,0])
    p01.scatter(sols[:,0], sols[:,1], color = "g")
    p01.scatter(survivors[:,0], survivors[:,1], color = "orange")

    p01.scatter(sol[0], sol[1])
    p23 = fig.add_subplot(gs[1,1])
    p23.scatter(sols[:,2], sols[:,3], color = "r")
    p23.scatter(survivors[:,2], survivors[:,3], color = "orange")
    p23.scatter(sol[2], sol[3])



