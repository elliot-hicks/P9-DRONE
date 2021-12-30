from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed = 1)

class CustomController(FlightController):

    def __init__(self):
        """
        Add params for the controller:
        """
        self.parameters = [0,0,0,0]
        self.k_y = self.parameters[0]  # constant
        self.b_y = self.parameters[1]  # drag coeff
        self.k_theta = self.parameters[2]
        self.b_theta = self.parameters[3]
        self.T_eq = 0.5  # equilibrium thrust
        self.thrust_angle = 0.3  # angle to move to, maybe tune this?
        self.pop_size = 4**4 # num of solutions
        self.number_generations = 1000

    def run_simulation(self, parameters):
        drone = self.init_drone()
        target = drone.get_next_target()  
        rewards = 0
        r_min = np.sqrt((drone.x-target[0])**2+(drone.y-target[1])**2) 
        for t in range(self.get_max_simulation_steps()): 
            drone.set_thrust(np.clip(self.get_thrusts(drone, parameters), 0, 1))
            target_hit = drone.step_simulation(self.get_time_interval())
            r = np.sqrt((drone.x-target[0])**2+(drone.y-target[1])**2)
            if target_hit:
                rewards+=100
                target = drone.get_next_target()  
                r_min = np.sqrt((drone.x-target[0])**2+(drone.y-target[1])**2) 
                print("HIT")
            elif r<r_min:
                rewards+=1/r # this line is the old version
                r_min = r
            else:
                rewards -=1/r
        return rewards, r_min
    
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
    
    def mutate(self, survivors):
        mutated_solutions = np.empty((0,4))
        for i in range(len(survivors[:,0])):
            mutations = np.empty((5,4))
            for g in range(4):
                mutations[:,g] = abs(np.random.normal(survivors[i,g],5, 5)) #   FIX VARIANCE
            mutated_solutions = np.vstack((mutated_solutions, mutations))
        return mutated_solutions
    
    def breed_survivors(self,survivors):
        crossover_solutions = self.crossover(survivors)
        mutated_solutions = self.mutate(survivors)     
        new_solutions = np.vstack((crossover_solutions, mutated_solutions))
        solutions = np.vstack((survivors, new_solutions))
        return solutions
    
    def select_survivors(self, solution_rewards, solutions):
        probabilities = np.array(solution_rewards - min(solution_rewards)).astype(float)       
        solution_inds = range(len(solutions[:,0]))
        if sum(probabilities) == 0:
            probabilities[:] = 1/self.pop_size
        else:
            probabilities = probabilities/sum(probabilities)
        # select 10% of population to create next generation

        survivor_inds = np.random.choice(solution_inds, int(self.pop_size/8),
                                         p = probabilities)
        survivors = np.clip(solutions[survivor_inds], 0, None)
        return survivors


    def train(self, number_of_episodes):
        #start by setting random parameters, for 100 solutions       
        k_values = np.linspace(0, 100, int(self.pop_size**0.25))
        b_values = np.linspace(0,10, int(self.pop_size**0.25))
        kk_x, bb_x, kk_theta, bb_theta = np.meshgrid(k_values, b_values, k_values, b_values)
        solutions = np.vstack((kk_x.flatten(), bb_x.flatten(), 
                               kk_theta.flatten(), bb_theta.flatten()))
        solutions = solutions.transpose() 
        # double up starting values to get kx, bx, k_theta, b_theta     
        average_rewards = []
        gen_closest_approaches = []
        for generation in range(number_of_episodes):#self.number_generations):
            print(generation)
            simulation_analysis = np.apply_along_axis(self.run_simulation, axis = 1, arr = solutions) 
            #sim analysis is a list of tuples (rewards, r_min)
            solution_rewards, closest_approaches = simulation_analysis[:,0], simulation_analysis[:,1]
            survivors = self.select_survivors(solution_rewards, solutions)
            solutions = self.breed_survivors(survivors)
            
            average_rewards.append(np.mean(solution_rewards))
            gen_closest_approaches.append(min(closest_approaches))
        
        print("training completed")
        return solutions, average_rewards, gen_closest_approaches
            
    def get_thrusts(self, drone: Drone, parameters) -> Tuple[float, float]:
        k_y, b_y = parameters[0], parameters[1]
        k_theta, b_theta = parameters[2], parameters[3]
        target = drone.get_next_target()
        dx = target[0] - drone.x
        dy = target[1] - drone.y
        
        if dx<0:
            thrust_angle = -self.thrust_angle
        else:
            thrust_angle = self.thrust_angle
        
        if abs(dx)>drone.game_target_size/5:
            T_eq = 1/(2*np.cos(drone.pitch))
            rotating_thrust = 1*k_theta*(thrust_angle - drone.pitch) - b_theta*drone.pitch_velocity#-5*drone.velocity_x/abs(dx)
            thrusts = [T_eq + rotating_thrust, T_eq - rotating_thrust] # CHANGE ME
            return np.clip(thrusts, 0, 1)
        else:
            T_eq = 0.5
            lifting_thrust = 1*k_y*(dy) - b_y*drone.velocity_y
            thrusts = [T_eq + lifting_thrust, T_eq + lifting_thrust]
            return np.clip(thrusts, 0, 1)
        

    
    def load(self):
        pass
    def save(self):
        pass
    
def run_for(episodes):
    c = CustomController()
    sols, gen_rs, rs = c.train(episodes)
    return gen_rs, sols, rs

"""
episodes = 10
gens, sols, rs = run_for(episodes)
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(gens, color = "purple")
ax3 = ax1.twinx()
ax3.plot(rs, color = "cyan")
ax2.scatter(sols[:,0], sols[:,1], color = "g")
ax2.scatter(sols[:,2], sols[:,3], color = "r")

sol = np.mean(sols, axis = 0)
"""