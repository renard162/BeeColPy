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
#     A modified scout bee for artificial bee colony algorithm and its performance
#     on optimization problems. Journal of King Saud University-Computer and Information
#     Sciences, 28(4), pp.395-406.
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
# [6] Mirjalili, S., Hashim, S., Taherzadeh, G., Mirjalili, S.Z. and Salehi,
#     S., 2011. A study of different transfer functions for binary version of
#     particle swarm optimization. In International Conference on Genetic and
#     Evolutionary Methods (Vol. 1, No. 1, pp. 2-7).
#     link: http://hdl.handle.net/10072/48831
#
# [7] Huang, S.C., 2015. Polygonal approximation using an artificial bee colony
#     algorithm. Mathematical Problems in Engineering, 2015.
#     doi: https://doi.org/10.1155/2015/375926
#
#------------------------------------------------------------------------------+
"""
import numpy as np
import random as rng
import warnings as wrn
# from scipy import special as sps #Only used in binary ABC form

class abc:
    """
    Class that applies Artificial Bee Colony (ABC) algorithm to find minimum
    or maximum of a function that's receive a vector of floats as input and
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
        initialization or during scout events. This option usually helps the
        algorithm stability because, in rare cases, NaN values can lock the
        algorithm in a infinite loop.
        Obs.: NaN protection can drastically increases calculation time if
              analysed function has too many values of domain returning NaN.


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
            - Number of complete iterations executed
            - Number of scout events during iterations
            - Number of times that NaN protection was activated

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
        self.nan_protection = nan_protection

        self.max_iterations = max([int(iterations), 1])
        if (iterations < 1):
            warn_message = 'Using the minimun value of iterations = 1'
            wrn.warn(warn_message, RuntimeWarning)
        
        self.employed_onlookers_count = max([int(colony_size/2), 2])
        if (colony_size < 4):
            warn_message = 'Using the minimun value of colony_size = 4'
            wrn.warn(warn_message, RuntimeWarning)
        
        if (scouts <= 0):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries))
            if (scouts < 0):
                warn_message = 'Negative scout count given, using default scout count: colony_size * dimension = ' + str(self.scout_limit)
                wrn.warn(warn_message, RuntimeWarning)
        elif (scouts < 1):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries) * scouts)
        else:
            self.scout_limit = int(scouts)

        self.scout_status = 0
        self.iteration_status = 0
        self.nan_status = 0

        self.foods = []
        for i in range(self.employed_onlookers_count):
            self.foods.append(_FoodSource(self,_ABCUtils(self)))
            _ABCUtils(self).nan_protection(i)

        try:
            self.best_food_source = self.foods[np.nanargmax([food.fit for food in self.foods])]
        except:
            self.best_food_source = self.foods[0]
            warn_message = 'All food sources\'s fit resulted in NaN and beecolpy can got stuck in an infinite loop during fit(). Enable nan_protection to prevent this.'
            wrn.warn(warn_message, RuntimeWarning)

        self.agents = []
        self.agents.append([food.position for food in self.foods])


    
    def fit(self):
        for iteration in range(1,self.max_iterations+1):
            #--> Employer bee phase <--
            #Generate and evaluate a neighbor point to every food source
            [_ABCUtils(self).food_source_dance(i) for i in range(self.employed_onlookers_count)]
            max_fit = np.nanmax([food.fit for food in self.foods])
            onlooker_probability = [_ABCUtils(self).prob_i(food.fit, max_fit) for food in self.foods]
            _ABCUtils(self).nan_lock_check()
    
            #--> Onlooker bee phase <--
            #Based in probability, generate a neighbor point and evaluate again some food sources
            #Same food source can be evaluated multiple times
            p = 0 #Onlooker bee index
            i = 0 #Food source index
            while (p < self.employed_onlookers_count):
                if (rng.uniform(0,1) <= onlooker_probability[i]):
                    p += 1
                    _ABCUtils(self).food_source_dance(i)
                    max_fit = np.nanmax([food.fit for food in self.foods])
                    if (self.foods[i].fit != max_fit):
                        onlooker_probability[i] = _ABCUtils(self).prob_i(self.foods[i].fit, max_fit)
                    else:
                        onlooker_probability = [_ABCUtils(self).prob_i(food.fit, max_fit) for food in self.foods]
                i = (i+1) if (i < (self.employed_onlookers_count-1)) else 0

            #--> Memorize best solution <--
            iteration_best_food_index = np.nanargmax([food.fit for food in self.foods])
            self.best_food_source = self.best_food_source if (self.best_food_source.fit >= self.foods[iteration_best_food_index].fit) \
                else self.foods[iteration_best_food_index]
            
            #--> Scout bee phase <--
            #Generate up to one new food source that does not improve over scout_limit evaluation tries
            trial_counters = [food.trial_counter for food in self.foods]
            if (max(trial_counters)>self.scout_limit):
                #Take the index of replaced food source
                trial_counters = np.where(np.array(trial_counters) == max(trial_counters))[0].tolist()
                i = trial_counters[rng.randrange(0,len(trial_counters))]
                
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
        return self.iteration_status, self.scout_status, self.nan_status



class binabc:
    """
    Class that applies Binary Artificial Bee Colony (BABC [5], based in Binary PSO [4])
    algorithm to find minimum or maximum of a function that's receive the number of
    bits as input and returns a vector of bits as output.


    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.
        Example: if the function is:
            def my_func(x): return x[0] or (x[1] and x[2])
            Put "my_func" as parameter.

    -=x=-
    bits_count : Int
        The number of bits that compose the output vector.

    boundaries : List of Tuples
        A list of tuples containing the lower and upper boundaries that will be
        applied over sigmoid function to determine the probability to bit become 1.
        Example: [(-5,5), (-20,20)]
    
    Obs.: If boundaries are set, then it's take the priority over the bits_count.
          If boundaries are not set, then the boundaries became (-10,10) to each bit.
    -=x=-

    [transfer_function] : String --optional-- (default: 'sigmoid')
        Defines the transfer function used to calculate the probability for each bit
        becomes '1'.
        The possibilities are explained on article [6]: http://hdl.handle.net/10072/48831
        If transfer_function = 'sigmoid' : S(x) = 1/[1 + exp(-x)]
        If transfer_function = 'sigmoid-2x' : S(x) = 1/[1 + exp(-2*x)]
        If transfer_function = 'sigmoid-x/2' : S(x) = 1/[1 + exp(-x/2)]
        If transfer_function = 'sigmoid-x/3' : S(x) = 1/[1 + exp(-x/3)]

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
        Obs.1: Scout_limit is rounded down in all cases.
        Obs.2: In Binary form, the scouts tends to be more relevant than in
               continuous form. If your problem are badly solved, try to reduce
               the scouts value.

    [iterations] : Int --optional-- (default: 50)
        The number of iterations executed by algorithm.

    [best_model_iterations] : int --optional-- (default:iterations count)
        Due stochastic aspect of Binary form of particle based metaheuristic,
        after execution of ABC, the cost function will be calculated
        "best_model_iterations" times and the "best" result will be returned.
        If best_model_iterations = 0 : Tries "iterations" times.
        If best_model_iterations = N : Tries "N" times.

    [min_max] : String --optional-- (default: 'min')
        Determines if algorithm will minimize or maximize the function.
        If min_max = 'min' : Try to localize the minimum of function.
        If min_max = 'max' : Try to localize the maximum of function.

    [nan_protection] : Int --optional-- (default: 4)
        If greater than 0, if the cost function returns NaN, the algorithm tries to
        recalculate the cost function up to "nan_protection" times.
        Obs.: NaN protection can drastically increases calculation time if
              analysed function has too many values of domain returning NaN.


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
            - Number of complete iterations executed
            - Number of scout events during iterations
            - Number of times that NaN protection was activated

    get_agents()
        Returns a list with the position of each food source during
        each iteration.

    """
    def __init__(self,
                 function,
                 bits_count: int=0,
                 boundaries: list=[],
                 transfer_function: str='sigmoid',
                 colony_size: int=40,
                 scouts: float=0.5,
                 iterations: int=50,
                 best_model_iterations: int=0,
                 min_max: str='min',
                 nan_protection: int=4):

        boundaries = [(-10,10) for _ in range(bits_count)] if (len(boundaries)==0) else boundaries
        self.function = function
        self.transfer_function = transfer_function
        self.best_model_iterations = iterations if (best_model_iterations<1) else best_model_iterations
        self.min_max_selector = min_max
        self.nan_protection = (nan_protection > 0)
        self.nan_count = int(nan_protection - 1)

        self.bin_abc_object = abc(_BinABCUtils(self).iteration_cost_function, boundaries,
                                  colony_size=colony_size, scouts=scouts, iterations=iterations,
                                  min_max=min_max, nan_protection=self.nan_protection)

        self.result_bit_vector = None

    def fit(self):
        self.bin_abc_object.fit()
        self.result_bit_vector = _BinABCUtils(self).get_best_solution(self.bin_abc_object.get_solution())
        return self.result_bit_vector

    def get_agents(self):
        return self.bin_abc_object.agents

    def get_solution(self):
        return self.result_bit_vector

    def get_status(self):
        return self.bin_abc_object.iteration_status, self.bin_abc_object.scout_status, self.bin_abc_object.nan_status



class amabc:
    """
    Class that applies Angle Modulated Artificial Bee Colony (AMABC [5])
    algorithm to find minimum or maximum of a function that's receive the number of
    bits as input and returns a vector of bits as output.


    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.
        Example: if the function is:
            def my_func(x): return x[0] or (x[1] and x[2])
            Put "my_func" as parameter.

    -=x=-
    bits_count : Int
        The number of bits that compose the output vector.

    boundaries : List of Tuples
        A list of tuples containing the lower and upper boundaries that will be
        applied over sigmoid function to determine the probability to bit become 1.
        Example: [(-5,5), (-20,20)]
    
    Obs.: If boundaries are set, then it's take the priority over the bits_count.
          If boundaries are not set, then the boundaries became (-2,2) to each bit.
    -=x=-

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
        Obs.1: Scout_limit is rounded down in all cases.
        Obs.2: In Binary form, the scouts tends to be more relevant than in
               continuous form. If your problem are badly solved, try to reduce
               the scouts value.

    [iterations] : Int --optional-- (default: 50)
        The number of iterations executed by algorithm.

    [min_max] : String --optional-- (default: 'min')
        Determines if algorithm will minimize or maximize the function.
        If min_max = 'min' : Try to localize the minimum of function.
        If min_max = 'max' : Try to localize the maximum of function.

    [nan_protection] : Boolean --optional-- (default: True)
        If true, re-generate food sources that get NaN value as cost during
        initialization or during scout events. This option usually helps the
        algorithm stability because, in rare cases, NaN values can lock the
        algorithm in a infinite loop.
        Obs.: NaN protection can drastically increases calculation time if
              analysed function has too many values of domain returning NaN.


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
            - Number of complete iterations executed
            - Number of scout events during iterations
            - Number of times that NaN protection was activated

    get_agents()
        Returns a list with the position of each food source during
        each iteration.

    """
    def __init__(self,
                 function,
                 bits_count: int=0,
                 boundaries: list=[],
                 colony_size: int=40,
                 scouts: float=0.5,
                 iterations: int=50,
                 min_max: str='min',
                 nan_protection: bool=True):

        boundaries = [(-2,2) for _ in range(bits_count)] if (len(boundaries)==0) else boundaries
        self.function = function

        self.am_abc_object = abc(_AMABCUtils(self).iteration_cost_function, boundaries,
                                  colony_size=colony_size, scouts=scouts, iterations=iterations,
                                  min_max=min_max, nan_protection=nan_protection)

        self.result_bit_vector = None

    def fit(self):
        self.am_abc_object.fit()
        self.result_bit_vector = _AMABCUtils(self).determine_bit_vector(self.am_abc_object.get_solution())
        return self.result_bit_vector

    def get_agents(self):
        return self.am_abc_object.agents

    def get_solution(self):
        return self.result_bit_vector

    def get_status(self):
        return self.am_abc_object.iteration_status, self.am_abc_object.scout_status, self.am_abc_object.nan_status




class _FoodSource:

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
        j = rng.randrange(0,len(self.abc.boundaries))
        
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

    def __init__(self,abc):
        self.abc = abc

    def nan_lock_check(self):
        if not(self.abc.nan_protection): #If NaN protection is anebled, there's no need to waste computational power checking NaN.
            if (sum([np.isnan(food.fit) for food in self.abc.foods]) >= len(self.abc.foods)):
                raise Exception('All food sources\'s fit resulted in NaN and beecolpy got stuck in an infinite loop. Enable nan_protection to prevent this.')

    def nan_protection(self,food_index):
        while (np.isnan(self.abc.foods[food_index].fit) and self.abc.nan_protection):
            self.abc.nan_status += 1
            self.abc.foods[food_index] = _FoodSource(self.abc,self)

    def prob_i(self,actual_fit,max_fit):
        # Improved probability function [7]
        return 0.9*(actual_fit/max_fit) + 0.1
        # Original probability function [1]
        # return actual_fit/np.sum([food.fit for food in self.abc.foods])

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
            d = int(rng.randrange(0,self.abc.employed_onlookers_count))
            if (d != index):
                break
        self.abc.foods[index].evaluate_neighbor(self.abc.foods[d].position)



class _BinABCUtils:

    def __init__(self, babc):
        self.babc = babc

    def sigmoid(self,x):
        return 1/(1 + np.exp((-1)*x))

    def transfer(self,probability): #Transfer functions discussion in [6]
        if (self.babc.transfer_function == 'sigmoid'):
            return self.sigmoid(probability)   #S(x) = 1/[1 + exp(-x)]
        elif (self.babc.transfer_function == 'sigmoid-2x'):
            return self.sigmoid(probability*2) #S(x) = 1/[1 + exp(-2*x)]
        elif (self.babc.transfer_function == 'sigmoid-x/2'):
            return self.sigmoid(probability/2) #S(x) = 1/[1 + exp(-x/2)]
        elif (self.babc.transfer_function == 'sigmoid-x/3'):
            return self.sigmoid(probability/3) #S(x) = 1/[1 + exp(-x/3)]
        else:
            raise Exception('\nInvalid transfer function. Valid values include:\n\'sigmoid\'\n\'sigmoid-2x\'\n\'sigmoid-x/2\'\n\'sigmoid-x/3\'')

    def determine_bit_vector(self,probability_vector):
        return [(rng.uniform(0,1) < self.transfer(probability)) for probability in probability_vector]

    def iteration_cost_function(self,probability_vector):
        cost = self.babc.function(self.determine_bit_vector(probability_vector))
        if self.babc.nan_protection:
            i = 0
            while (np.isnan(cost) and (i < self.babc.nan_count)):
                cost = self.babc.function(self.determine_bit_vector(probability_vector))
                i += 1
        return cost

    def get_best_solution(self,probability_vector):
        for i in range(self.babc.best_model_iterations):
            temp_bit_vector = self.determine_bit_vector(probability_vector)
            temp_cost_value = self.babc.function(temp_bit_vector)
            if self.babc.nan_protection:
                j = 0
                while (np.isnan(temp_cost_value) and (j < self.babc.nan_count)):
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



class _AMABCUtils:

    def __init__(self, amabc):
        self.amabc = amabc

    def angle_modulation(self,angle):
        # Equation (8) from [5] with constants:
        # g(x) = sin{2 * pi * (x-a) * b * cos[2 * pi * (x-a) * c]} + d
        # a=0 b=1 c=1 d=0
        pi2 = 2*np.pi
        return np.sin(pi2 * angle * np.cos(pi2 * angle))

    def determine_bit_vector(self,angle_vector):
        return [(self.angle_modulation(angle) > 0) for angle in angle_vector]

    def iteration_cost_function(self,angle_vector):
        return self.amabc.function(self.determine_bit_vector(angle_vector))
