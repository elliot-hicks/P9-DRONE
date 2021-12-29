from flight_controller import FlightController
from drone import Drone
from typing import Tuple
import numpy as np

class CustomController(FlightController):

    def __init__(self):
        """
        Add params for the controller:
        """
        self.k = 0  # constant
        self.b = 0  # drag coeff
        self.T_eq = 0.5  # equilibrium thrust
        self.thrust_angle = 0.1  # angle to move to, maybe tune this?
        self.delta_thrust = 0.1
        self.pop_size = 100 # num of solutions
        self.number_generations = 1000

    def run_simulation(self, parameters, drone):
        rewards = 0
        for t in range(self.get_max_simulation_steps()):
            drone.set_thrust(self.get_thrusts(drone))
            drone.step_simulation(self.get_time_interval())
            # get rewards
        return rewards
    
    def crossover(self, survivors):
        # crossover random pairs of survivors
        crossover_solutions = np.empty((0,2))
        solution_inds = range(len(survivors[:,0]))
        for i in solution_inds:
            random_pair = np.random.choice(solution_inds, 2, replace = False)
            pair_crossover = np.empty((3,2))
            sol_1 = survivors[random_pair[0],:]
            sol_2 = survivors[random_pair[1],:]        
            # full x-over
            pair_crossover[0,:] = [(sol_1[0] + sol_2[0])/2, (sol_1[1] + sol_2[1])/2]
            # k x-over
            pair_crossover[1,:] = [(sol_1[0] + sol_2[0])/2, sol_1[1]]
            # b x-over
            pair_crossover[2,:] = [sol_1[0], (sol_1[1] + sol_2[1])/2]
            crossover_solutions = np.vstack(crossover_solutions, pair_crossover)
        return crossover_solutions
    
    def mutate(self, survivors):
        mutated_solutions = np.empty((0,2))
        for i in range(len(survivors[:,0])):
            mutations = np.empty((10,2))
            mutations[:,0] = np.random.normal(survivors[i,0], 0.1*survivors[i,0], 6)
            mutations[:,1] = np.random.normal(survivors[i,1], 0.1*survivors[i,1], 6)
            mutated_solutions = np.vstack(mutated_solutions, mutations)
        return mutated_solutions
    
    def breed_survivors(self,survivors):
        crossover_solutions = self.crossover(survivors)
        mutated_solutions = self.mutate(survivors)     
        new_solutions = np.vstack(crossover_solutions, mutated_solutions)
        solutions = np.vstack(survivors, new_solutions)
        return solutions
    
    def select_survivors(self, solution_rewards, solutions):
        probabilities = solution_rewards - min(solution_rewards)
        try:
            probabilities /=sum(probabilities)
        except ValueError: # guard against zero division
            probabilities = 1/self.pop_size
            
        # select 10% of population to create next generation
        survivors = np.random.choice(solutions, int(0.1*self.pop_size), probabilities)
        return survivors
        
    def train(self, drone):
        #start by setting random parameters, for 100 solutions
        solutions = np.zeros((100, 2)) # solutions are (k,b)
        solutions[:, 0] = np.random.uniform(0, 1000, 100)
        solutions[:, 1] = np.random.uniform(0,10,100)
        for generation in range(self.number_generations):
            solution_rewards = np.apply_along_axis(self.run_simulation, 
                                                   axis = 1, arr = solutions,
                                                   drone = drone)      
            survivors = self.select_survivors(solution_rewards, solutions)
            solutions = self.breed_solutions(survivors)
             
            
    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        return (0.5, 0.5) # Replace this with your custom algorithm
    def load(self):
        pass
    def save(self):
        pass
    
    
c = CustomController()
drone = c.init_drone()
c.train(drone)