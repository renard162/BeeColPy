'''
#----------------------------------------------------------------------+
#
#   Samuel C P Oliveira
#   samuelcpoliveira@gmail.com
#   Artificial Bee Colony Optimization
#   This project is licensed under the MIT License.
#
#----------------------------------------------------------------------+
#
# Bibliography
#
#    [1] Karaboga, D. and Basturk, B., 2007
#        A powerful and efficient algorithm for numerical function 
#        optimization: artificial bee colony (ABC) algorithm. Journal 
#        of global optimization, 39(3), pp.459-471. 
#        DOI: https://doi.org/10.1007/s10898-007-9149-x
#
#    [2] Liu, T., Zhang, L. and Zhang, J., 2013
#        Study of binary artificial bee colony algorithm based on 
#        particle swarm optimization. Journal of Computational 
#        Information Systems, 9(16), pp.6459-6466. 
#        Link: https://api.semanticscholar.org/CorpusID:8789571
#
#    [3] Anuar, S., Selamat, A. and Sallehuddin, R., 2016
#        A modified scout bee for artificial bee colony algorithm and 
#        its performance on optimization problems. Journal of King Saud 
#        University-Computer and Information Sciences, 28(4), 
#        pp.395-406. 
#        DOI: https://doi.org/10.1016/j.jksuci.2016.03.001
#
#    [4] Kennedy, J. and Eberhart, R.C., 1997
#        A discrete binary version of the particle swarm algorithm. 
#        In 1997 IEEE International conference on systems, man, and 
#        cybernetics. Computational cybernetics and simulation 
#        (Vol. 5, pp. 4104-4108). IEEE. 
#        DOI: https://doi.org/10.1109/ICSMC.1997.637339
#
#    [5] Pampará, G. and Engelbrecht, A.P., 2011
#        Binary artificial bee colony optimization. In 2011 IEEE 
#        Symposium on Swarm Intelligence (pp. 1-8). IEEE. 
#        DOI: https://doi.org/10.1109/SIS.2011.5952562
#
#    [6] Mirjalili, S., Hashim, S., Taherzadeh, G., Mirjalili, S.Z. 
#    and Salehi, S., 2011
#        A study of different transfer functions for binary version 
#        of particle swarm optimization. In International Conference 
#        on Genetic and Evolutionary Methods (Vol. 1, No. 1, pp. 2-7). 
#        Link: http://hdl.handle.net/10072/48831
#
#    [7] Huang, S.C., 2015
#        Polygonal approximation using an artificial bee colony 
#        algorithm. Mathematical Problems in Engineering, 2015. 
#        DOI: https://doi.org/10.1155/2015/375926
#
#----------------------------------------------------------------------+
'''
import numpy as np
import random as rng
import warnings as wrn
from collections import Counter


class abc:
    '''
    Class that applies Artificial Bee Colony (ABC) algorithm to find 
    minimum or maximum of a function that's receive a vector of floats 
    as input and returns a float as output.

    https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm

    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.
        Example: if the function is:
            def my_func(x): return x[0]**2 + x[1]**2 + 5*x[1]
            
            Put "my_func" as parameter.


    boundaries : List of Tuples
        A list of tuples containing the lower and upper boundaries of 
        each dimension of function domain.
        
        Obs.: The number of boundaries determines the dimension of 
        function.

        Example: [(-5,5), (-20,20)]


    [colony_size] : Int --optional-- (default: 40)
        A value that determines the number of bees in algorithm. Half 
        of this amount determines the number of points analyzed (food 
        sources).
        
        According articles, half of this number determines the amount 
        of Employed bees and other half is Onlooker bees.


    [scouts] : Float --optional-- (default: 0.5)
        Determines the limit of tries for scout bee discard a food 
        source and replace for a new one.
            - If scouts = 0 : 
                Scout_limit = colony_size * dimension

            - If scouts = (0 to 1) : 
                Scout_limit = colony_size * dimension * scouts
                    Obs.: scouts = 0.5 is used in [3] as benchmark.

            - If scouts >= 1 : 
                Scout_limit = scouts

        Obs.: Scout_limit is rounded down in all cases.


    [iterations] : Int --optional-- (default: 50)
        The number of iterations executed by algorithm.


    [min_max] : String --optional-- (default: 'min')
        Determines if algorithm will minimize or maximize the function.
            - If min_max = 'min' : 
                Locate the minimum of function.

            - If min_max = 'max' : 
                Locate the maximum of function.


    [nan_protection] : Boolean --optional-- (default: True)
        If true, re-generate food sources that get NaN value as cost 
        during initialization or during scout events. This option 
        usually helps the algorithm stability because, in rare cases, 
        NaN values can lock the algorithm in a infinite loop.
        
        Obs.: NaN protection can drastically increases calculation 
        time if analysed function has too many values of domain 
        returning NaN.


    Methods
    ----------
    fit()
        Execute the algorithm with defined parameters.

        Obs.: Returns a list with values found as minimum/maximum 
        coordinate.


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


    Bibliography
    ----------
    [1] Karaboga, D. and Basturk, B., 2007
        A powerful and efficient algorithm for numerical function 
        optimization: artificial bee colony (ABC) algorithm. Journal 
        of global optimization, 39(3), pp.459-471. 
        DOI: https://doi.org/10.1007/s10898-007-9149-x

    [2] Liu, T., Zhang, L. and Zhang, J., 2013
        Study of binary artificial bee colony algorithm based on 
        particle swarm optimization. Journal of Computational 
        Information Systems, 9(16), pp.6459-6466. 
        Link: https://api.semanticscholar.org/CorpusID:8789571

    [3] Anuar, S., Selamat, A. and Sallehuddin, R., 2016
        A modified scout bee for artificial bee colony algorithm and 
        its performance on optimization problems. Journal of King Saud 
        University-Computer and Information Sciences, 28(4), 
        pp.395-406. 
        DOI: https://doi.org/10.1016/j.jksuci.2016.03.001

    '''
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

        self.max_iterations = int(max([iterations, 1]))
        if (iterations < 1):
            warn_message = 'Using the minimun value of iterations = 1'
            wrn.warn(warn_message, RuntimeWarning)
        
        self.employed_onlookers_count = int(max([(colony_size/2), 2]))
        if (colony_size < 4):
            warn_message = 'Using the minimun value of colony_size = 4'
            wrn.warn(warn_message, RuntimeWarning)
        
        if (scouts <= 0):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries))
            if (scouts < 0):
                warn_message = 'Negative scout count given, using default scout ' \
                    'count: colony_size * dimension = ' + str(self.scout_limit)
                wrn.warn(warn_message, RuntimeWarning)
        elif (scouts < 1):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries) * scouts)
        else:
            self.scout_limit = int(scouts)

        self.scout_status = 0
        self.iteration_status = 0
        self.nan_status = 0

        self.foods = [None] * self.employed_onlookers_count
        for i in range(len(self.foods)):
            _ABC_engine(self).generate_food_source(i)

        try:
            self.best_food_source = self.foods[np.nanargmax([food.fit for food in self.foods])]
        except:
            self.best_food_source = self.foods[0]
            warn_message = 'All food sources\'s fit resulted in NaN and beecolpy can got stuck ' \
                         'in an infinite loop during fit(). Enable nan_protection to prevent this.'
            wrn.warn(warn_message, RuntimeWarning)

        self.agents = []


    def fit(self):
        '''
        Execute the algorithm with defined parameters.

        Obs.: Returns a list with values found as minimum/maximum 
        coordinate.
        '''
        self.agents = []
        self.agents.append([food.position for food in self.foods])

        for iteration in range(1, self.max_iterations+1):
            #--> Employer bee phase <--
            #Generate and evaluate a neighbor point to every food source
            _ABC_engine(self).employer_bee_phase()
    
            #--> Onlooker bee phase <--
            #Based in probability, generate a neighbor point and evaluate again some food sources
            #Same food source can be evaluated multiple times
            _ABC_engine(self).onlooker_bee_phase()
            
            #--> Memorize best solution <--
            _ABC_engine(self).memorize_best_solution()

            #--> Scout bee phase <--
            #Generate up to one new food source that does not improve over scout_limit evaluation tries
            _ABC_engine(self).scout_bee_phase()
            
            self.agents.append([food.position for food in self.foods])
            self.iteration_status = iteration
        
        return self.best_food_source.position


    def get_agents(self):
        '''
        Returns a list with the position of each food source during
        each iteration.
        '''
        return self.agents


    def get_solution(self):
        '''
        Returns the value obtained after fit() the method.

        Obs.: If fit() is not executed, return the position of
        best initial condition.
        '''
        return self.best_food_source.position


    def get_status(self):
        '''
        Returns a tuple with:
            - Number of complete iterations executed
            - Number of scout events during iterations
            - Number of times that NaN protection was activated
        '''
        return self.iteration_status, \
               self.scout_status, \
               self.nan_status




class bin_abc:
    '''
    Class that applies Artificial Bee Colony in a binary domain 
    function to find minimum or maximum of a function that's receive 
    the number of bits as input and returns a vector of bits as output.
    
    There two methods in this solver:
        - Angle Modulated Artificial Bee Colony (AMABC [5]):
            A deterministic based solver. (default)

        - Binary Artificial Bee Colony (BABC [5], based in BPSO [4]):
            A stochastic based solver.


    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.

        Example: if the function is:
            def my_func(x): return x[0] or (x[1] and x[2])
            
            Put "my_func" as parameter.


    -=x=--=x=--=x=--=x=--=x=-

    bits_count : Int
        The number of bits that compose the output vector.


    boundaries : List of Tuples
        A list of tuples containing the lower and upper boundaries 
        that will be applied over sigmoid function to determine the 
        probability to bit become 1.

        Example: [(-5,5), (-20,20)]
    
    Obs.:
        - If boundaries are set: 
            boundaries take the priority over the bits_count.

        - If boundaries are not set: 
            boundaries became (-2,2) to each bit in AMABC method or 
            (-10,10) to each bit in BABC method.
    
    -=x=--=x=--=x=--=x=--=x=-


    [method] : String --optional-- (default: 'am')
        Select the applied solver:
            - If method = 'am' : 
                Applied Angle Modulated ABC (AMABC).

            - If method = 'bin' : 
                Applied Binary ABC (BABC).


    [colony_size] : Int --optional-- (default: 40)
        A value that determines the number of bees in algorithm. Half 
        of this amount determines the number of points analyzed 
        (food sources).
        
        According articles, half of this number determines the amount 
        of Employed bees and other half is Onlooker bees.


    [scouts] : Float --optional-- (default: 0.5)
        Determines the limit of tries for scout bee discard a food 
        source and replace for a new one.
            - If scouts = 0 : 
                Scout_limit = colony_size * dimension

            - If scouts = (0 to 1) : 
                Scout_limit = colony_size * dimension * scouts
                    Obs.: scouts = 0.5 is used in [3] as benchmark.

            - If scouts >= 1 : 
                Scout_limit = scouts

        Obs.1: Scout_limit is rounded down in all cases.
        
        Obs.2: In Binary form, the scouts tends to be more relevant 
        than in continuous form. If your problem are badly solved, 
        try to reduce the scouts value.


    [iterations] : Int --optional-- (default: 50)
        The number of iterations executed by algorithm.


    [min_max] : String --optional-- (default: 'min')
        Determines if algorithm will minimize or maximize the function.
            - If min_max = 'min' : 
                Locate the minimum of function.

            - If min_max = 'max' : 
                Locate the maximum of function.


    [nan_protection] : Boolean or Int --optional-- 
    (default (boolean): True)
        With "method='am'", this variable are used as a boolean.
        
        With "method='bin'", this variable determines the number of 
        times the function are recalculated when it returns a NaN. 
        (default (int): 3)
        
        If true or greater than 0, re-generate food sources that get 
        NaN value as cost during initialization or during scout 
        events. This option usually helps the algorithm stability 
        because, in rare cases, NaN values can lock the algorithm in 
        a infinite loop.
        
        Obs.: NaN protection can drastically increases calculation 
        time if analysed function has too many values of domain 
        returning NaN.


    [transfer_function] : String --optional-- (default: 'sigmoid')
        Only used with "method='bin'". Defines the transfer function 
        used to calculate the probability for each bit becomes '1'.
        
        The possibilities are explained on article [6]:
            - If transfer_function = 'sigmoid' : 
                S(x) = 1/(1 + exp(-x))

            - If transfer_function = 'sigmoid-2x' : 
                S(x) = 1/(1 + exp(-2*x))

            - If transfer_function = 'sigmoid-x/2' : 
                S(x) = 1/(1 + exp(-x/2))

            - If transfer_function = 'sigmoid-x/3' : 
                S(x) = 1/(1 + exp(-x/3))


    [result_format] : String --optional-- (default: 'average')
        Only used with "method='bin'". In a stochastic method, the 
        result vector are represented by a probability vector with
        the probability of each bit becomes "True". This property 
        determines how output bit vector will be estimated.
            - If result_format = 'average' :
                Returns the most frequent bit vector after 
                "best_model_iterations" simulations of the probability
                vector. This approach is ideal to solve problems with
                highly random elements.

            - If result_format = 'best' :
                Returns the best result after "best_model_iterations" 
                simulations of the probability vector. This approach is 
                useful to solve highly noisy problems.


    [best_model_iterations] : int --optional-- 
    (default: iterations count)
        Only used with "method='bin'". Due stochastic aspect of 
        Binary form of particle based metaheuristic, after execution 
        of ABC, the cost function will be calculated 
        "best_model_iterations" times and the "best" or the "most 
        frequent" result will be returned.
            - If best_model_iterations = 0 : (default)
                Tries "iterations" times. 

            - If best_model_iterations = N : 
                Tries "N" times.

            Obs.: If "best_model_iterations" (or "iterations") is even, 
            then "best_model_iterations" is increased by one.


    Methods
    ----------
    fit()
        Execute the algorithm with defined parameters.

        Obs.: Returns a list with values found as minimum/maximum 
        coordinate.


    get_solution()
        Returns the value obtained after fit() the method.

        Obs.: If fit() is not executed, return "None".

        Parameters
        ----------
        [probability_vector] : bool --optional-- (default: False)
            Only used with "method='bin'". Returns the vector with 
            probability of each bit becomes "True". Useful to use 
            probability as component of stopping criteria or to 
            evaluate solution stability.
                - If probability_vector = True :
                    "get_solution" returns a vector with the 
                    probability of each bit becomes "True".

                - If probability_vector = False: (default)
                    "get_solution" returns the solution bit vector.


    get_status()
        Returns a tuple with:
            - Number of complete iterations executed
            - Number of scout events during iterations
            - Number of times that NaN protection was activated


    get_agents()
        Returns a list with the position of each food source during
        each iteration.

        Obs.: In binary form, this method returns the position of 
        each food source after transformation "binary -> continuous". 
        I.e. returns the values applied on angle modulation function 
        in AMABC or the values applied on transfer function in BABC.
        

    Bibliography
    ----------
    [1] Karaboga, D. and Basturk, B., 2007
        A powerful and efficient algorithm for numerical function 
        optimization: artificial bee colony (ABC) algorithm. Journal 
        of global optimization, 39(3), pp.459-471. 
        DOI: https://doi.org/10.1007/s10898-007-9149-x

    [2] Liu, T., Zhang, L. and Zhang, J., 2013
        Study of binary artificial bee colony algorithm based on 
        particle swarm optimization. Journal of Computational 
        Information Systems, 9(16), pp.6459-6466. 
        Link: https://api.semanticscholar.org/CorpusID:8789571

    [3] Anuar, S., Selamat, A. and Sallehuddin, R., 2016
        A modified scout bee for artificial bee colony algorithm and 
        its performance on optimization problems. Journal of King Saud 
        University-Computer and Information Sciences, 28(4), 
        pp.395-406. 
        DOI: https://doi.org/10.1016/j.jksuci.2016.03.001

    [4] Kennedy, J. and Eberhart, R.C., 1997
        A discrete binary version of the particle swarm algorithm. 
        In 1997 IEEE International conference on systems, man, and 
        cybernetics. Computational cybernetics and simulation 
        (Vol. 5, pp. 4104-4108). IEEE. 
        DOI: https://doi.org/10.1109/ICSMC.1997.637339

    [5] Pampará, G. and Engelbrecht, A.P., 2011
        Binary artificial bee colony optimization. In 2011 IEEE 
        Symposium on Swarm Intelligence (pp. 1-8). IEEE. 
        DOI: https://doi.org/10.1109/SIS.2011.5952562

    [6] Mirjalili, S., Hashim, S., Taherzadeh, G., Mirjalili, S.Z. 
    and Salehi, S., 2011
        A study of different transfer functions for binary version 
        of particle swarm optimization. In International Conference 
        on Genetic and Evolutionary Methods (Vol. 1, No. 1, pp. 2-7). 
        Link: http://hdl.handle.net/10072/48831

    [7] Huang, S.C., 2015
        Polygonal approximation using an artificial bee colony 
        algorithm. Mathematical Problems in Engineering, 2015. 
        DOI: https://doi.org/10.1155/2015/375926

    '''
    def __init__(self,
                 function,
                 bits_count: int=0,
                 boundaries: list=[],
                 colony_size: int=40,
                 scouts: float=0.5,
                 iterations: int=50,
                 min_max: str='min',
                 method: str='am',
                 nan_protection: bool=True,
                 transfer_function: str='sigmoid',
                 result_format: str='average',
                 best_model_iterations: int=0):
        
        self.method = method
        self.function = function
        
        bits_count = int(bits_count)
        if ((len(boundaries) == 0) and (bits_count <= 0)):
            raise Exception('\nInvalid bit vector length. ' \
                            '\'bits_count\' need to be greater than 0 or define \'boundaries\' list.')

        if (nan_protection < 0):
            warn_message = 'NaN protection disabled. Negative nan_protection given.'
            wrn.warn(warn_message, RuntimeWarning)
        self._nan_protection = (nan_protection > 0)

        self.result_bit_vector = None

        self.executed_fit = False
        
        #Method selector
        if (self.method == 'am'): #Angle Modulated
            boundaries = [(-2, 2) for _ in range(bits_count)] if (len(boundaries) == 0) \
                                                              else boundaries

            self._bin_abc_object = abc(_AMABC_engine(self).am_cost_function, boundaries,
                                       colony_size = colony_size,
                                       scouts = scouts,
                                       iterations = iterations,
                                       min_max = min_max,
                                       nan_protection = self._nan_protection)

        elif (self.method == 'bin'): #Binary ABC
            self.transfer_function = transfer_function
            self.min_max_selector = min_max

            self.result_format = result_format
            _BABC_engine(self).check_result_format()

            best_model_iterations = best_model_iterations if (best_model_iterations > 0) \
                                                          else iterations
            best_model_iterations += int(not(best_model_iterations % 2)) #Grants the odd value
            if (best_model_iterations < 3):
                warn_message = 'Using best_model_iterations = 3'
                wrn.warn(warn_message, RuntimeWarning)
            self.best_model_iterations = int(max([best_model_iterations, 3]))

            self._nan_count = 2 if (isinstance(nan_protection, bool) and (self._nan_protection)) \
                                else int(max([(nan_protection - 1), 0]))

            self.boundaries = [(-10, 10) for _ in range(bits_count)] if (len(boundaries) == 0) \
                                                                     else boundaries

            self._bin_abc_object = abc(_BABC_engine(self).bin_cost_function, self.boundaries,
                                       colony_size = colony_size,
                                       scouts = scouts,
                                       iterations = iterations,
                                       min_max = min_max,
                                       nan_protection = self._nan_protection)

        else:
            raise Exception('\nInvalid method. Valid values include:\n\'am\'\n\'bin\'')
        

    def fit(self):
        '''
        Execute the algorithm with defined parameters.

        Obs.: Returns a list with values found as minimum/maximum 
        coordinate.
        '''
        self._bin_abc_object.fit()
        self.executed_fit = True
        if (self.method == 'am'): #Angle Modulated
            self.result_bit_vector = _AMABC_engine(self).get_bit_vector(
                                                            self._bin_abc_object.get_solution())
        elif (self.method == 'bin'): #Binary ABC
            self.result_bit_vector = _BABC_engine(self).get_result_vector(
                                                            self._bin_abc_object.get_solution())
        return self.result_bit_vector


    def get_agents(self):
        '''
        Returns a list with the position of each food source during
        each iteration.

        Obs.: In binary form, this method returns the position of 
        each food source after transformation "binary -> continuous". 
        I.e. returns the values applied on angle modulation function 
        in AMABC or the values applied on transfer function in BABC.
        '''
        return self._bin_abc_object.agents


    def get_solution(self, probability_vector: bool=False):
        '''
        Returns the value obtained after fit() the method.

        Obs.: If fit() is not executed, return "None".

        Parameters
        ----------
        [probability_vector] : bool --optional-- (default: False)
            Only used with "method='bin'". Returns the vector with 
            probability of each bit becomes "True". Useful to use 
            probability as component of stopping criteria or to 
            evaluate solution stability.
                - If probability_vector = True :
                    "get_solution" returns a vector with the 
                    probability of each bit becomes "True".

                - If probability_vector = False: (default)
                    "get_solution" returns the solution bit vector.
        '''
        if (probability_vector and (self.method == 'bin') and self.executed_fit):
            return _BABC_engine(self).get_probability_vector(
                                            self._bin_abc_object.get_solution())
        else:
            return self.result_bit_vector


    def get_status(self):
        '''
        Returns a tuple with:
            - Number of complete iterations executed
            - Number of scout events during iterations
            - Number of times that NaN protection was activated
        '''
        return self._bin_abc_object.iteration_status, \
               self._bin_abc_object.scout_status, \
               self._bin_abc_object.nan_status



class binabc(bin_abc):
    '''
    DEPRECATION WARNING:
        This function will be removed in next versions. 
        Use "bin_abc" with "method='bin'" and "result_format='best'" 
        instead.
        
    Class that applies Binary Artificial Bee Colony (BABC [5], based 
    in Binary PSO [4]) algorithm to find minimum or maximum of a 
    function that's receive the number of bits as input and returns 
    a vector of bits as output.
    '''
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
        
        warn_message = 'This object will be removed. ' \
            'Use "bin_abc" with "method=\'bin\'" instead.'
        wrn.warn(warn_message, DeprecationWarning)

        bin_abc.__init__(self,
                         function = function,
                         bits_count = bits_count,
                         boundaries = boundaries,
                         colony_size = colony_size,
                         scouts = scouts,
                         iterations = iterations,
                         min_max = min_max,
                         method = 'bin',
                         nan_protection = nan_protection,
                         transfer_function = transfer_function,
                         result_format = 'best',
                         best_model_iterations = best_model_iterations)



class amabc(bin_abc):
    '''
    DEPRECATION WARNING:
        This function will be removed in next versions. 
        Use "bin_abc" with "method='am'" instead.
        
    Class that applies Angle Modulated Artificial Bee Colony 
    (AMABC [5]) algorithm to find minimum or maximum of a function 
    that's receive the number of bits as input and returns a vector 
    of bits as output.
    '''
    def __init__(self,
                 function,
                 bits_count: int=0,
                 boundaries: list=[],
                 colony_size: int=40,
                 scouts: float=0.5,
                 iterations: int=50,
                 min_max: str='min',
                 nan_protection: bool=True):

        warn_message = 'This object will be removed. ' \
            'Use "bin_abc" with "method=\'am\'" instead.'
        wrn.warn(warn_message, DeprecationWarning)

        bin_abc.__init__(self,
                         function = function,
                         bits_count = bits_count,
                         boundaries = boundaries,
                         colony_size = colony_size,
                         scouts = scouts,
                         iterations = iterations,
                         min_max = min_max,
                         method = 'am',
                         nan_protection = nan_protection,
                         transfer_function = 'sigmoid',
                         result_format = 'average',
                         best_model_iterations = 0)




class _FoodSource:

    def __init__(self, abc, engine):
        #When a food source is initialized, randomize a position inside boundaries and calculate the "fit"
        self.abc = abc
        self.engine = engine
        self.trial_counter = 0
        self.position = [rng.uniform(*self.abc.boundaries[i]) for i in range(len(self.abc.boundaries))]
        self.fit = self.engine.calculate_fit(self.position)


    def evaluate_neighbor(self, partner_position):
        #Randomize one coodinate (one dimension) to generate a neighbor point
        j = rng.randrange(0, len(self.abc.boundaries))
        
        #eq. (2.2) [1] (new coordinate "x_j" to generate a neighbor point)
        xj_new = self.position[j] + rng.uniform(-1, 1)*(self.position[j] - partner_position[j])

        #Check boundaries
        xj_new = self.abc.boundaries[j][0] if (xj_new < self.abc.boundaries[j][0]) else \
            self.abc.boundaries[j][1] if (xj_new > self.abc.boundaries[j][1]) else xj_new
        
        #Changes the coordinate "j" from food source to new "x_j" generating the neighbor point
        neighbor_position = [(self.position[i] if (i != j) else xj_new) for i in range(len(self.abc.boundaries))]
        neighbor_fit = self.engine.calculate_fit(neighbor_position)

        #Greedy selection
        if (neighbor_fit > self.fit):
            self.position = neighbor_position
            self.fit = neighbor_fit
            self.trial_counter = 0
        else:
            self.trial_counter += 1



class _ABC_engine:

    def __init__(self, abc):
        self.abc = abc


    def check_nan_lock(self):
        if not(self.abc.nan_protection):
            if np.all([np.isnan(food.fit) for food in self.abc.foods]):
                raise Exception('All food sources\'s fit resulted in NaN and beecolpy got ' \
                                'stuck in an infinite loop. Enable nan_protection to prevent this.')


    def execute_nan_protection(self, food_index):
        while (np.isnan(self.abc.foods[food_index].fit) and self.abc.nan_protection):
            self.abc.nan_status += 1
            self.abc.foods[food_index] = _FoodSource(self.abc, self)


    def prob_i(self, actual_fit, max_fit):
        # Improved probability function [7]
        return 0.9*(actual_fit/max_fit) + 0.1
        # Original probability function [1]
        # return actual_fit/np.sum([food.fit for food in self.abc.foods])


    def calculate_fit(self, evaluated_position):
        #eq. (2) [2] (Convert "cost function" to "fit function")
        cost = self.abc.cost_function(evaluated_position)
        if (self.abc.min_max_selector == 'min'): #Minimize function
            fit_value = (1 + np.abs(cost)) if (cost < 0) else (1/(1 + cost))
        else: #Maximize function
            fit_value = (1 + cost) if (cost > 0) else (1/(1 + np.abs(cost)))
        return fit_value


    def food_source_dance(self, index):
        #Generate a partner food source to generate a neighbor point to evaluate
        while True: #Criterion from [1] geting another food source at random
            d = int(rng.randrange(0, self.abc.employed_onlookers_count))
            if (d != index):
                break
        self.abc.foods[index].evaluate_neighbor(self.abc.foods[d].position)


    def generate_food_source(self, index):
        self.abc.foods[index] = _FoodSource(self.abc, self)
        self.execute_nan_protection(index)


    def employer_bee_phase(self):
        #Generate and evaluate a neighbor point to every food source
        for i in range(self.abc.employed_onlookers_count):
            self.food_source_dance(i)


    def onlooker_bee_phase(self):
        #Based in probability, generate a neighbor point and evaluate again some food sources
        #Same food source can be evaluated multiple times
        max_fit = np.nanmax([food.fit for food in self.abc.foods])
        onlooker_probability = [self.prob_i(food.fit, max_fit) for food in self.abc.foods]
        self.check_nan_lock()
        p = 0 #Onlooker bee index
        i = 0 #Food source index
        while (p < self.abc.employed_onlookers_count):
            if (rng.uniform(0, 1) <= onlooker_probability[i]):
                p += 1
                self.food_source_dance(i)
                self.check_nan_lock()
                max_fit = np.nanmax([food.fit for food in self.abc.foods])
                if (self.abc.foods[i].fit != max_fit):
                    onlooker_probability[i] = self.prob_i(self.abc.foods[i].fit, max_fit)
                else:
                    onlooker_probability = [self.prob_i(food.fit, max_fit) for food in self.abc.foods]
            i = (i+1) if (i < (self.abc.employed_onlookers_count-1)) else 0


    def scout_bee_phase(self):
        #Generate up to one new food source that does not improve over scout_limit evaluation tries
        trial_counters = [food.trial_counter for food in self.abc.foods]
        if (max(trial_counters) > self.abc.scout_limit):
            #Take the index of replaced food source
            trial_counters = np.where(np.array(trial_counters) == max(trial_counters))[0].tolist()
            i = trial_counters[rng.randrange(0, len(trial_counters))]
            self.generate_food_source(i) #Replace food source
            self.abc.scout_status += 1


    def memorize_best_solution(self):
        best_food_index = np.nanargmax([food.fit for food in self.abc.foods])
        if (self.abc.foods[best_food_index].fit >= self.abc.best_food_source.fit):
            self.abc.best_food_source = self.abc.foods[best_food_index]




class _BABC_engine:

    def __init__(self, babc):
        self.babc = babc


    def check_result_format(self):
        valid_formats = ['average', 'best']
        if (self.babc.result_format not in valid_formats):
            raise Exception('\nInvalid result format. Valid values include:' \
                            '\n\'average\'\n\'best\'')


    def sigmoid(self, x):
        return 1/(1 + np.exp((-1)*x))


    def transfer(self, value): #Transfer functions discused in [6]
        if (self.babc.transfer_function == 'sigmoid'):
            return self.sigmoid(value)   #S(x) = 1/(1 + exp(-x))
        elif (self.babc.transfer_function == 'sigmoid-2x'):
            return self.sigmoid(value*2) #S(x) = 1/(1 + exp(-2*x))
        elif (self.babc.transfer_function == 'sigmoid-x/2'):
            return self.sigmoid(value/2) #S(x) = 1/(1 + exp(-x/2))
        elif (self.babc.transfer_function == 'sigmoid-x/3'):
            return self.sigmoid(value/3) #S(x) = 1/(1 + exp(-x/3))
        else:
            raise Exception('\nInvalid transfer function. Valid values include:' \
                '\n\'sigmoid\'\n\'sigmoid-2x\'\n\'sigmoid-x/2\'\n\'sigmoid-x/3\'')


    def get_probability_vector(self, value_vector):
        return [self.transfer(value) for value in value_vector]


    def get_bit_vector(self, value_vector):
        probability_vector = self.get_probability_vector(value_vector)
        return [(rng.uniform(0, 1) < probability) for probability in probability_vector]


    def recalculate_nan(self, bit_vector, cost_value, value_vector):
        j = 0
        while (np.isnan(cost_value) and (j < self.babc._nan_count)):
            bit_vector = self.get_bit_vector(value_vector)
            cost_value = self.babc.function(bit_vector)
            j += 1
        return bit_vector, cost_value


    def bin_cost_function(self, value_vector):
        bit_vector = self.get_bit_vector(value_vector)
        cost_value = self.babc.function(bit_vector)

        if self.babc._nan_protection:
            _, cost_value = self.recalculate_nan(bit_vector, cost_value, value_vector)
            
        return cost_value


    def get_best_model(self, value_vector):
        cost_value = np.nan
        for i in range(self.babc.best_model_iterations):
            temp_bit_vector = self.get_bit_vector(value_vector)
            temp_cost_value = self.babc.function(temp_bit_vector)

            if self.babc._nan_protection:
                temp_bit_vector, temp_cost_value = self.recalculate_nan(
                                                    temp_bit_vector, temp_cost_value, value_vector)

            if (self.babc.min_max_selector == 'min'): #Minimize function
                if ((i < 1) or (temp_cost_value < cost_value)):
                    cost_value = temp_cost_value
                    bit_vector = temp_bit_vector
            else: #Maximize function
                if ((i < 1) or (temp_cost_value > cost_value)):
                    cost_value = temp_cost_value
                    bit_vector = temp_bit_vector
        return bit_vector


    def get_average_model(self, value_vector):
        solution_collection = np.zeros((self.babc.best_model_iterations, 
                                        len(self.babc.boundaries)))
        solution_collection.fill(np.nan)
        bit_vector = [None] * len(self.babc.boundaries)

        for i in range(self.babc.best_model_iterations):
            temp_bit_vector = self.get_bit_vector(value_vector)

            if self.babc._nan_protection:
                temp_cost_value = self.babc.function(temp_bit_vector)
                temp_bit_vector, _ = self.recalculate_nan(
                                        temp_bit_vector, temp_cost_value, value_vector)
            
            solution_collection[i,:] = temp_bit_vector

        for j in range(len(self.babc.boundaries)):
            simulated_bits = solution_collection[:,j]
            bit_vector[j] = bool(Counter(simulated_bits).most_common(1)[0][0])
        return bit_vector


    def get_result_vector(self, value_vector):
        self.check_result_format()

        if (self.babc.result_format == 'average'):
            return self.get_average_model(value_vector)
        elif (self.babc.result_format == 'best'):
            return self.get_best_model(value_vector)




class _AMABC_engine:

    def __init__(self, amabc):
        self.amabc = amabc


    def angle_modulation(self, angle):
        # Equation (8) from [5] with constants:
        # g(x) = sin{2 * pi * (x-a) * b * cos[2 * pi * (x-a) * c]} + d
        # a=0 b=1 c=1 d=0
        PI2 = 2 * np.pi
        return np.sin(PI2 * angle * np.cos(PI2 * angle))


    def get_bit_vector(self, angle_vector):
        return [(self.angle_modulation(angle) > 0) for angle in angle_vector]


    def am_cost_function(self, angle_vector):
        return self.amabc.function(self.get_bit_vector(angle_vector))
