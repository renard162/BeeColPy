**BeeColPy**
============

BeeColPy is a module for function optimization through artificial bee colony
algorithm, a method developed by Karaboga [1], a variant of classical
particle swarm optimization.



**Websites:**

> Source code: https://github.com/renard162/BeeColPy/
>
> Introduction: https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm
>



**Installation**
------------

**Dependencies**

> BeeColPy requires:
>
> - Python (>= 3.0)
> - NumPy (>= 1.1.0)
>
> **BeeColPy do not support Python 2.7.**

**User installation**

~~~~~~~~~~~~~~~~~
pip install beecolpy
~~~~~~~~~~~~~~~~~



**Usage Instructions**
----------

**For cost functions with continuous domain:**

~~~~~~~~~~~~~~~~~python
#Step-by-step:
#Create object and set the solver parameters:
abc_obj = abc(function,
              boundaries,
              colony_size=40,
              scouts=0.5,
              iterations=50,
              min_max='min',
              nan_protection=True,
              log_agents=True)

#Execute algorithm: 
abc_obj.fit()

#Get solution obtained after fit() execution:
solution = abc_obj.get_solution()

"""
    Obs.: Each time fit() was executed, the algorithm iterate 
    'iterations' times resuming from last fit() execution.
"""

"""
    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.
        Example: if the function is:
            def my_func(x): return x[0]**2 + x[1]**2 + 5*x[1]
            
            Use "my_func" as parameter.


    boundaries : List of Tuples
        A list of tuples containing the lower and upper boundaries of 
        each dimension of function domain.

        Obs.: The number of boundaries determines the dimension of 
        function.

        Example: A function F(x1, x2) = y with:
            (-5 <= x1 <= 5) and (-20 <= x2 <= 20) have the boundaries:
                [(-5,5), (-20,20)]


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
            - If min_max = 'min' : (default)
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


    [log_agents] : Boolean --optional-- (default: False)
        If true, beecolpy will register, before each iteration, the
        position of each food source. Useful to debug but, if there a
        high amount of food sources and/or iterations, this option
        drastically increases memory usage.


    [seed] : Int --optional-- (default: None)
        If defined as an int, set the seed used in all random process.


    Methods
    ----------
    fit()
        Execute the algorithm with defined parameters.

        Obs.: Returns a list with values found as minimum/maximum 
        coordinate.


    get_solution()
        Returns the value obtained after fit() the method.


    get_status()
        Returns a tuple with:
            - Number of complete iterations executed
            - Number of scout events during iterations
            - Number of times that NaN protection was activated


    get_agents()
        Returns a list with the position of each food source during
        each iteration if "log_agents = True".

        Parameters
        ----------
        [reset_agents] : bool --optional-- (default: False)
            If true, the food source position log will be cleaned in
            next fit().
"""
~~~~~~~~~~~~~~~~~



**For cost function with binary domain:**

~~~~~~~~~~~~~~~~~python
#Step-by-step:
#Create object and set the solver parameters:
bin_abc_obj = bin_abc(function,
                      bits_count,
                      method='am',
                      colony_size=40,
                      scouts=0.5,
                      iterations=50,
                      best_model_iterations=0,
                      min_max='min',
                      nan_protection=True,
                      transfer_function='sigmoid',
                      best_model_iterations=0,
                      log_agents=True)

#Execute algorithm: 
bin_abc_obj.fit()

#Get solution after execute fit() without execute it again:
solution = bin_abc_obj.get_solution()

"""
    Obs.: Each time fit() was executed, the algorithm iterate 
    'iterations' times resuming from last fit() execution.
"""

"""
    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.

        Example: if the function is:
            def my_func(x): return x[0] or (x[1] and x[2])

            Use "my_func" as parameter.


    -=x=--=x=--=x=--=x=--=x=-

    Just one of these parameters are mandatory. If you don't know 
    exactly how binary solvers work, just inform the number of bits 
    (bits_count) and the default boundaries will be used. These 
    boundaries usually are enough to solve most problems.

    bits_count : Int
        The number of bits that compose the output vector.


    boundaries : List of Tuples
        A list of tuples containing the lower and upper boundaries 
        that will be applied over sigmoid or angle modulation function 
        to determine the probability to bit become 1.

        Example: A function F(b1, b2) = y with:
            (-5 <= b1 <= 5) and (-20 <= b2 <= 20) have the boundaries:
                [(-5,5), (-20,20)]

    Obs.:
        - If boundaries are set: 
            boundaries take the priority over the bits_count.

        - If boundaries are not set: 
            boundaries became (-2,2) to each bit in AMABC method or 
            (-10,10) to each bit in BABC method.

    -=x=--=x=--=x=--=x=--=x=-


    [method] : String --optional-- (default: 'am')
        Select the applied solver:
            - If method = 'am' : (default)
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
            - If min_max = 'min' : (default)
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
            - If transfer_function = 'sigmoid' : (default)
                S(x) = 1/(1 + exp(-x))

            - If transfer_function = 'sigmoid-2x' : 
                S(x) = 1/(1 + exp(-2*x))

            - If transfer_function = 'sigmoid-x/2' : 
                S(x) = 1/(1 + exp(-x/2))

            - If transfer_function = 'sigmoid-x/3' : 
                S(x) = 1/(1 + exp(-x/3))


    [result_format] : String --optional-- (default: 'best')
        Only used with "method='bin'". In a stochastic method, the 
        result vector are represented by a probability vector with
        the probability of each bit becomes "True". This property 
        determines how output bit vector will be estimated.
            - If result_format = 'average' :
                Returns the most frequent bit vector after 
                "best_model_iterations" simulations of the probability
                vector. This approach is ideal to solve problems with
                highly random elements.
                    Obs.: To use this method efficiently, use high 
                    values in "best_model_iterations". Usually values 
                    greater than 100 have better results.

            - If result_format = 'best' : (default)
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


    [log_agents] : Boolean --optional-- (default: False)
        If true, beecolpy will register, before each iteration, the
        position of each food source. Useful to debug but, if there a
        high amount of food sources and/or iterations, this option
        drastically increases memory usage.


    [seed] : Int --optional-- (default: None)
        If defined as an int, set the seed used in all random process.


    Methods
    ----------
    fit()
        Execute the algorithm with defined parameters.

        Obs.: Returns a list with values found as minimum/maximum 
        coordinate.


    get_solution()
        Returns the value obtained after fit() the method.

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
        each iteration if "log_agents = True".

        Obs.: In binary form, this method returns the position of 
        each food source after transformation "binary -> continuous". 
        I.e. returns the values applied on angle modulation function 
        in AMABC or the values applied on transfer function in BABC.

        Parameters
        ----------
        [reset_agents] : bool --optional-- (default: False)
            If true, the food source position log will be cleaned in
            next fit().
"""
~~~~~~~~~~~~~~~~~



**Example**
----------

~~~~~~~~~~~~~~~~~python
"""
To find the minimum  of sphere function on interval (-10 to 10) with
2 dimensions in domain using default parameters:
"""

from beecolpy import abc

def sphere(x):
	total = 0
	for i in range(len(x)):
		total += x[i]**2
	return total
	
abc_obj = abc(sphere, [(-10,10), (-10,10)]) #Load data
abc_obj.fit() #Execute the algorithm

#If you want to get the obtained solution after execute the fit() method:
solution = abc_obj.get_solution()

#If you want to get the number of iterations executed, number of times that
#scout event occur and number of times that NaN protection actuated:
iterations = abc_obj.get_status()[0]
scout = abc_obj.get_status()[1]
nan_events = abc_obj.get_status()[2]

#If you want to get a list with position of all points (food sources) used in each iteration:
food_sources = abc_obj.get_agents()

~~~~~~~~~~~~~~~~~



**Author**
--------------

**Samuel Carlos Pessoa Oliveira** - samuelcpoliveira@gmail.com



**License**
--------------

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.



**Bibliography**
---------------

[1] Karaboga, D. and Basturk, B., 2007 A powerful and efficient algorithm for numerical	function optimization: artificial bee colony ABC) algorithm. Journal of global optimization, 39(3), pp.459-471. Doi: https://doi.org/10.1007/s10898-007-9149-x

[2] Liu, T., Zhang, L. and Zhang, J., 2013 Study of binary artificial bee colony algorithm based on particle swarm optimization. Journal of Computational Information Systems, 9(16), pp.6459-6466. Link: https://api.semanticscholar.org/CorpusID:8789571

[3] Anuar, S., Selamat, A. and Sallehuddin, R., 2016 A modified scout bee for artificial bee colony algorithm and its performance on optimization problems. Journal of King Saud University-Computer and Information Sciences, 28(4), pp. 95-406. Doi: https://doi.org/10.1016/j.jksuci.2016.03.001
	
[4] Kennedy, J. and Eberhart, R.C., 1997, October. A discrete binary version of the particle swarm algorithm. In 1997 IEEE International conference on systems, man, and cybernetics. Computational cybernetics and simulation (Vol. 5, pp. 4104-4108). IEEE. Doi: https://doi.org/10.1109/ICSMC.1997.637339

[5] Pampará, G. and Engelbrecht, A.P., 2011, April. Binary artificial bee colony optimization. In 2011 IEEE Symposium on Swarm Intelligence (pp. 1-8). IEEE. Doi: https://doi.org/10.1109/SIS.2011.5952562

[6] Mirjalili, S., Hashim, S., Taherzadeh, G., Mirjalili, S.Z. and Salehi, S., 2011. A study of different transfer functions for binary version of particle swarm optimization. In International Conference on Genetic and Evolutionary Methods (Vol. 1, No. 1, pp. 2-7). Link: http://hdl.handle.net/10072/48831

[7] Huang, S.C., 2015. Polygonal approximation using an artificial bee colony algorithm. Mathematical Problems in Engineering, 2015. Doi: https://doi.org/10.1155/2015/375926
