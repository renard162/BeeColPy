"""
#------------------------------------------------------------------------------+
#
#   Samuel C P Oliveira
#   samuelcpoliveira@gmail.com
#   Artificial Bee Colony Optimization
#   This project is licensed under the MIT License.
#   v1.0 (April 2020)
#
#------------------------------------------------------------------------------+
#
# Bibliography
#
# [1] Karaboga, D. and Basturk, B., 2007
#     A powerful and efficient algorithm for numerical function optimization:
#     artificial bee colony (ABC) algorithm. Journal of global optimization, 39(3), pp.459-471.
#     doi: https://doi.org/10.1007/s10898-007-9149-x
# 
# [2] Liu, T., Zhang, L. and Zhang, J., 2013
#     Study of binary artificial bee colony algorithm based on particle swarm optimization.
#     Journal of Computational Information Systems, 9(16), pp.6459-6466.
#     link: https://api.semanticscholar.org/CorpusID:8789571
#
# [3] Anuar, S., Selamat, A. and Sallehuddin, R., 2016
#     A modified scout bee for artificial bee colony algorithm and its performance on optimization
#     problems. Journal of King Saud University-Computer and Information Sciences, 28(4), pp.395-406.
#     doi: https://doi.org/10.1016/j.jksuci.2016.03.001
#
# [4] Kennedy, J. and Eberhart, R.C., 1997, October. A discrete binary version
#     of the particle swarm algorithm. In 1997 IEEE International conference on
#     systems, man, and cybernetics. Computational cybernetics and simulation
#     (Vol. 5, pp. 4104-4108). IEEE.
#     doi: https://doi.org/10.1109/ICSMC.1997.637339
#
# [5] Pampará, G. and Engelbrecht, A.P., 2011, April. Binary artificial bee
#     colony optimization. In 2011 IEEE Symposium on Swarm Intelligence
#     (pp. 1-8). IEEE.
#     doi: https://doi.org/10.1109/SIS.2011.5952562
#
#------------------------------------------------------------------------------+
"""

# %%
import numpy as np
import numpy.random as rng
import scipy.special as sps #Only used in binary ABC form

class abc:
    """
    Class that applies Artificial Bee Colony (ABC) algorithm to find minimun
    or maximun of a function thats receive a vector of floats as input and
    returns a float as output.


    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.
        Example: if the function is:
            def my_func(x): return x[0]**2 + x[1]**2 + 5*x[1]
            Put "my_func" as parameter.

    boundaries : List of Tuples
        A list of tuples containing the lower and upper boundaries of each
        dimension of function domain.
        Obs.: The number of boundaries determines the dimension of function.
        Example: [(-5,5), (-20,20)]

    [colony_size] : Int --optional-- (default: 40)
        A value that determines the number of bees in algorithm. Half of this
        amount determines the number of points analyzed (food sources).
        According articles, half of this number determines the amount of
        Employed bees and other half is Onlooker bees.

    [scouts] : Float --optional-- (default: 0.5)
        Determines the limit of tries for scout bee discard a food source and
        replace for a new one.
        If scouts = 0 : Scout_limit = colony_size * dimension
        If scouts = (0 to 1) : Scout_limit = colony_size * dimension * scouts
            Obs.: scout = 0.5 is used in [3] as benchmark.
        If scout = (1 to iterations) : Scout_limit = scout
        If scout >= iterations: Scout event never occurs
        Obs.: Scout_limit is rounded down in all cases.

    [iterations] : Int --optional-- (default: 50)
        The number of iterations executed by algorithm.

    [min_max] : String --optional-- (default: 'min')
        Determines if algorithm will minimize or maximize the function.
        If min_max = 'min' : Try to localize the minimum of function.
        If min_max = 'max' : Try to localize the maximum of function.

    [nan_protection] : Boolean --optional-- (default: True)
        If true, re-generate food sources that get NaN value as cost during
        initialization or during scout events. This option usualy helps the
        algorith stability because, in rare cases, NaN values can lock the
        algorithm in a infinite loop.


    Methods
    ----------
    fit()
        Execute the algorithm with defined parameters.
        Obs.: Returns a list with values found as minimum/maximum coordinate.

    get_solution()
        Returns the value obtained after fit() the method.
        Obs.: If fit() is not executed, return the position of
              best initial condition.

    get_status()
        Returns a tuple with:
            - Number of iterations executed
            - Number of scout events during iterations

    get_agents()
        Returns a list with the position of each food source during
        each iteration.

    """
    def __init__(self,
                 function,
                 boundaries,
                 colony_size: int=40,
                 scouts: float=0.5,
                 iterations: int=50,
                 min_max: str='min',
                 nan_protection: bool=True):

        self.boundaries = boundaries
        self.min_max_selector = min_max
        self.cost_function = function
        self.max_iterations = int(iterations)
        self.nan_protection = nan_protection
        
        self.employed_onlookers_count = int(colony_size/2)
        
        if (scouts == 0):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries))
        elif (scouts < 1):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries) * scouts)
        else:
            self.scout_limit = int(scouts)
        
        self.foods = []
        for i in range(self.employed_onlookers_count):
            self.foods.append(_FoodSource(self,_ABCUtils(self)))
            _ABCUtils(self).nan_protection(i)
        
        self.best_food_source = self.foods[np.argmax([food.fit for food in self.foods])]
        self.agents = []
        self.agents.append([food.position for food in self.foods])
        
        self.scout_status = 0
        self.iteration_status = 0

    
    def fit(self):
        for iteration in range(1,self.max_iterations+1):
            #--> Employer bee phase <--
            #Generate and evaluate a neighbor point to every food source
            [_ABCUtils(self).food_source_dance(i) for i in range(self.employed_onlookers_count)]
            max_fit = np.max([food.fit for food in self.foods])
            onlooker_probability = [0.9*(food.fit/max_fit)+0.1 for food in self.foods]
    
            #--> Onlooker bee phase <--
            #Based in probability, generate a neighbor point and evaluate again some food sources
            #Same food source can be evaluated multiple times
            p = 0 #Onlooker bee index
            i = 0 #Food source index
            while (p < self.employed_onlookers_count):
                if (rng.uniform(0,1) < onlooker_probability[i]):
                    p += 1
                    _ABCUtils(self).food_source_dance(i)
                    max_fit = np.max([food.fit for food in self.foods])
                    onlooker_probability = [0.9*(food.fit/max_fit)+0.1 for food in self.foods]
                i = (i+1) if (i < (self.employed_onlookers_count-1)) else 0

            #--> Memorize best solution <--
            iteration_best_food_index = np.argmax([food.fit for food in self.foods])
            self.best_food_source = self.best_food_source if (self.best_food_source.fit >= self.foods[iteration_best_food_index].fit) \
                else self.foods[iteration_best_food_index]
            
            #--> Scout bee phase <--
            #Generate up to one new food source that does not improve over scout_limit evaluation tries
            trial_counters = [food.trial_counter for food in self.foods]
            if (max(trial_counters)>self.scout_limit):
                #Take the index of replaced food source
                trial_counters = np.where(np.array(trial_counters) == max(trial_counters))[0].tolist()
                i = trial_counters[rng.randint(0,len(trial_counters))]
                
                self.foods[i] = _FoodSource(self,_ABCUtils(self)) #Replace food source
                _ABCUtils(self).nan_protection(i)
                self.scout_status += 1
            
            self.agents.append([food.position for food in self.foods])
            self.iteration_status = iteration
        return self.best_food_source.position

    def get_agents(self):
        return self.agents
    
    def get_solution(self):
        return self.best_food_source.position
    
    def get_status(self):
        return self.iteration_status, self.scout_status



class binabc:
    """
    Class that applies Binary Artificial Bee Colony (BABC) algorithm to find minimun
    or maximun of a function thats receive the number of bits as input and
    returns a vector of bits as output.


    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.
        Example: if the function is:
            def my_func(x): return x[0] or (x[1] and x[2])
            Put "my_func" as parameter.

    bits_count : Int
        The number of bits that compose the output vector.

    # boundaries : List of Tuples
    #     A list of tuples containing the lower and upper boundaries of each
    #     dimension of function domain.
    #     Obs.: The number of boundaries determines the dimension of function.
    #     Example: [(-5,5), (-20,20)]

    [colony_size] : Int --optional-- (default: 40)
        A value that determines the number of bees in algorithm. Half of this
        amount determines the number of points analyzed (food sources).
        According articles, half of this number determines the amount of
        Employed bees and other half is Onlooker bees.

    [scouts] : Float --optional-- (default: 0.5)
        Determines the limit of tries for scout bee discard a food source and
        replace for a new one.
        If scouts = 0 : Scout_limit = colony_size * dimension
        If scouts = (0 to 1) : Scout_limit = colony_size * dimension * scouts
            Obs.: scout = 0.5 is used in [3] as benchmark.
        If scout = (1 to iterations) : Scout_limit = scout
        If scout >= iterations: Scout event never occurs
        Obs.: Scout_limit is rounded down in all cases.

    [iterations] : Int --optional-- (default: 50)
        The number of iterations executed by algorithm.

    [min_max] : String --optional-- (default: 'min')
        Determines if algorithm will minimize or maximize the function.
        If min_max = 'min' : Try to localize the minimum of function.
        If min_max = 'max' : Try to localize the maximum of function.

    [nan_protection] : Boolean --optional-- (default: True)
        If true, re-generate food sources that get NaN value as cost during
        initialization or during scout events. This option usualy helps the
        algorith stability because, in rare cases, NaN values can lock the
        algorithm in a infinite loop.


    Methods
    ----------
    fit()
        Execute the algorithm with defined parameters.
        Obs.: Returns a list with values found as minimum/maximum coordinate.

    get_solution()
        Returns the value obtained after fit() the method.
        Obs.: If fit() is not executed, return "None"

    get_status()
        Returns a tuple with:
            - Number of iterations executed
            - Number of scout events during iterations

    get_agents()
        Returns a list with the position of each food source during
        each iteration.

    """
    def __init__(self,
                 function,
                 bits_count,
                 boundaries: list=[],
                 colony_size: int=40,
                 scouts: float=0.5,
                 iterations: int=50,
                 best_model_iterations: int=0,
                 min_max: str='min',
                 nan_protection: int=2):

        boundaries = [(-10,10) for _ in range(bits_count)] if (len(boundaries)==0) else boundaries
        self.function = function
        self.best_model_iterations = iterations if (best_model_iterations<1) else best_model_iterations
        self.min_max_selector = min_max
        self.nan_protection = nan_protection

        self.bin_abc_object = abc(_BinABCUtils(self).iteration_cost_function, boundaries,
                                  colony_size=colony_size, scouts=scouts, iterations=iterations,
                                  min_max=min_max, nan_protection=(nan_protection>0))

        self.result_bit_vector = None

    def fit(self):
        self.bin_abc_object.fit()
        self.result_bit_vector = _BinABCUtils(self).get_medium_result(self.bin_abc_object.get_solution())
        return self.result_bit_vector

    def get_agents(self):
        return self.bin_abc_object.get_agents()

    def get_solution(self):
        return self.result_bit_vector

    def get_status(self):
        return self.bin_abc_object.iteration_status, self.bin_abc_object.scout_status




class _FoodSource:
    """
    Class that represents a food source (evaluated point). Useful to developers.


    Parameters
    ----------
    abc : Class
        A main class with variables and methods thats correlate food sources.

    utils : Class
        A class with methods invisible to user.


    Methods
    ----------
    evaluate_neighbor(evaluated_position = Position of evaluated point)
        Using evaluated position (partner position), generate a neighbor point,
        evaluate the "fit" value of it and applies greedy selection on it.
        If the "fit" value of neighbor point is better, permute this position with
        original food source's position and set trial counter of this food source
        to 0.
        If original food source "fit" value is better, mantain the original position
        and increases trial counter in 1.
        Trial counter is used to trigger scout event during iterations.
    """
    def __init__(self,abc,utils):
        #When a food source is initialized, randomize a position inside boundaries and
        #calculate the "fit"
        self.abc = abc
        self.abcu = utils
        self.trial_counter = 0
        self.position = [rng.uniform(*self.abc.boundaries[i]) for i in range(len(self.abc.boundaries))]
        self.fit = self.abcu.calculate_fit(self.position)
        
    def evaluate_neighbor(self,partner_position):
        #Randomize one coodinate (one dimension) to generate a neighbor point
        j = rng.randint(0,len(self.abc.boundaries))
        
        #eq. (2.2) [1] (new coordinate "x_j" to generate a neighbor point)
        xj_new = self.position[j] + rng.uniform(-1,1)*(self.position[j] - partner_position[j])

        #Check boundaries
        xj_new = self.abc.boundaries[j][0] if (xj_new < self.abc.boundaries[j][0]) else \
            self.abc.boundaries[j][1] if (xj_new > self.abc.boundaries[j][1]) else xj_new
        
        #Changes the coordinate "j" from food source to new "x_j" generating the neighbor point
        neighbor_position = [(self.position[i] if (i != j) else xj_new) for i in range(len(self.abc.boundaries))]
        neighbor_fit = self.abcu.calculate_fit(neighbor_position)

        #Greedy selection
        if (neighbor_fit > self.fit):
            self.position = neighbor_position
            self.fit = neighbor_fit
            self.trial_counter = 0
        else:
            self.trial_counter += 1



class _ABCUtils:
    """
    Class with methods invisible to user. 


    Parameters
    ----------
    abc : Class
        A main class with variables and methods thats correlate food sources.


    Methods
    ----------
    nan_protection(food_index = index of tested food source)
        Re-generate food sources that get NaN as cost value to prevent loop lock.

    calculate_fit(evaluated_position = Position of evaluated point)
        Calculate the cost function (value of function in point) and
        returns the "fit" value associated (float) (according [2]).

    food_source_dance(index = index (int) of food source target)
        Randomizes a "neighbor" point to evaluate the <index> food source.
    """
    def __init__(self,abc):
        self.abc = abc

    def nan_protection(self,food_index):
        while (np.isnan(self.abc.foods[food_index].fit) and self.abc.nan_protection):
            self.abc.foods[food_index] = _FoodSource(self.abc,self)

    def calculate_fit(self,evaluated_position):
        #eq. (2) [2] (Convert "cost function" to "fit function")
        cost = self.abc.cost_function(evaluated_position)
        if (self.abc.min_max_selector == 'min'): #Minimize function
            fit_value = (1 + np.abs(cost)) if (cost < 0) else (1/(1 + cost))
        else: #Maximize function
            fit_value = (1 + cost) if (cost > 0) else (1/(1 + np.abs(cost)))
        return fit_value
    
    def food_source_dance(self,index):
        #Generate a partner food source to generate a neighbor point to evaluate
        while True: #Criterion from [1] geting another food source at random
            d = int(rng.randint(0,self.abc.employed_onlookers_count))
            if (d != index):
                break
        self.abc.foods[index].evaluate_neighbor(self.abc.foods[d].position)



class _BinABCUtils:
    def __init__(self, babc):
        self.babc = babc

    def determine_bit_vector(self,probability_vector):
        return [(rng.uniform(0,1) < sps.expit(probability)) for probability in probability_vector]

    def iteration_cost_function(self,probability_vector):
        cost = self.babc.function(self.determine_bit_vector(probability_vector))
        if (self.babc.nan_protection > 0):
            i = 0
            while (np.isnan(cost) and (i < (self.babc.nan_protection-1))):
                cost = self.babc.function(self.determine_bit_vector(probability_vector))
                i += 1
        return cost

    def get_medium_result(self,probability_vector):
        for i in range(self.babc.best_model_iterations):
            temp_bit_vector = self.determine_bit_vector(probability_vector)
            temp_cost_value = self.babc.function(temp_bit_vector)
            if (self.babc.nan_protection > 0):
                j = 0
                while (np.isnan(temp_cost_value) and (j < (self.babc.nan_protection-1))):
                    temp_bit_vector = self.determine_bit_vector(probability_vector)
                    temp_cost_value = self.babc.function(temp_bit_vector)
                    j += 1
            if (self.babc.min_max_selector == 'min'): #Minimize function
                if ((i<1) or (temp_cost_value < cost_value)):
                    cost_value = temp_cost_value
                    bit_vector = temp_bit_vector
            else: #Maximize function
                if ((i<1) or (temp_cost_value > cost_value)):
                    cost_value = temp_cost_value
                    bit_vector = temp_bit_vector
        return bit_vector
