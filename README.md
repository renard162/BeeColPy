**BeeColPy**
============

BeeColPy is a module for function optimization through artificial bee colony
algorithm, a method developed by Karaboga [1], a variant of classical
particle swarm optimization.

Website: https://github.com/renard162/BeeColPy/

**Installation**
------------

**Dependencies**
~~~~~~~~~~~~~~~~~

BeeColPy requires:

- Python (>= 3.0)
- NumPy (>= 1.1.0)

**BeeColPy do not support Python 2.7.**
BeeColPy needs Python 3.0 or newer.
~~~~~~~~~~~~~~~~~

**User installation**

If you already have a working installation of numpy,
the easiest way to install BeeColPy is using ``pip``:

~~~~~~~~~~~~~~~~~
    pip install beecolpy
~~~~~~~~~~~~~~~~~

**Usage Instructions**
----------
For cost functions with continuous domain:
~~~~~~~~~~~~~~~~~
Class that applies Artificial Bee Colony (ABC) algorithm to find minimum
or maximum of a function that's receive a vector of floats as input and
returns a float as output.

Step-by-step:
#Load data and options:
abc_obj = abc(function, boundaries, colony_size=40, scouts=0.5,
			 iterations=50, min_max='min', nan_protection=True)

#Execute algorithm: 
abc_obj.fit()

#Get solution after execute fit() without execute it again:
solution = abc_obj.get_solution()

"""
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
	Obs.: NaN protection can lock algorithm in infinite loop if the function has
			too many values of domain returning NaN.


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
		- Number of iterations executed
		- Number of scout events during iterations
		- Number of times that NaN protection was activated

get_agents()
	Returns a list with the position of each food source during each iteration.
"""
~~~~~~~~~~~~~~~~~


For cost function with binary domain using probabilistic-based solver.
~~~~~~~~~~~~~~~~~
Class that applies Binary Artificial Bee Colony (BABC [5], based in Binary PSO [4])
algorithm to find minimum or maximum of a function that's receive the number of
bits as input and returns a vector of bits as output.

Step-by-step:
#Load data and options:
abc_obj = abc(function, bits_count, transfer_function='sigmoid',
			  colony_size=40, scouts=0.5, iterations=50,
			  best_model_iterations=0,
			  min_max='min', nan_protection=4)

#Execute algorithm: 
abc_obj.fit()

#Get solution after execute fit() without execute it again:
solution = abc_obj.get_solution()

"""
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
	Obs.: NaN protection can lock algorithm in infinite loop if the function has
			too many values of domain returning NaN.


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
		- Number of times that NaN protection was activated

get_agents()
	Returns a list with the position of each food source during
	each iteration.

"""
~~~~~~~~~~~~~~~~~


For cost function with binary domain using deterministic-based solver.
~~~~~~~~~~~~~~~~~
Class that applies Angle Modulated Artificial Bee Colony (AMABC [5])
algorithm to find minimum or maximum of a function that's receive the number of
bits as input and returns a vector of bits as output.

Step-by-step:
#Load data and options:
abc_obj = abc(function, bits_count, colony_size=40, scouts=0.5, iterations=50,
			  min_max='min', nan_protection=True)

#Execute algorithm:
abc_obj.fit()

#Get solution after execute fit() without execute it again:
solution = abc_obj.get_solution()

"""
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
	Obs.: NaN protection can lock algorithm in infinite loop if the function has
			too many values of domain returning NaN.


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
		- Number of times that NaN protection was activated

get_agents()
	Returns a list with the position of each food source during
	each iteration.

"""
~~~~~~~~~~~~~~~~~

**Example**
----------
~~~~~~~~~~~~~~~~~
"""
To find the minimum  of sphere function on interval (-10 to 10) with
2 dimensions in domain using default options:
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

#If you want to get the number of iterations executed and number of times that
#scout event occur:
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