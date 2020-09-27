# %%
from beecolpy import *

from copy import deepcopy
import math as mt
import numpy as np
import numpy.random as rng
import numpy.testing as npt

def sphere(x): #Continuous benchmark
    #Sphere function
    #Min: x=(0,0)
    total = 0
    for i in range(len(x)):
        total += x[i]**2
    return total

def translate_bin(b):
    return np.sum([(b[::-1][i])*mt.pow(2,i) for i in range(len(b))])

def squared_bin(b): #Binary benchmark
    #y=(x-1)*(x-3)*(x-11)
    #y = x^3 - 15 x^2 + 47 x - 33
    #Min: x=0 b=[0000 0000] (Local)
    #     x=8.055 b~[0000 1000] (Global)
    x = translate_bin(b)
    return mt.pow(x,3) - 15*mt.pow(x,2) + 47*(x) - 33

rng.seed(0)
base_abc_obj = abc(sphere, [(-10,10) for _ in range(2)],
                   colony_size=10, scouts=0.5,
                   iterations=10, min_max='min',
                   nan_protection=True)

base_bin_abc_obj = binabc(squared_bin, bits_count=4,
                          transfer_function='sigmoid',
                          colony_size=10, scouts=0.5,
                          iterations=10, min_max='min',
                          nan_protection=3)

# %%
def test_food_source_generation():
    global base_abc_obj
    # return [food.position for food in base_abc_obj.foods]
    npt.assert_array_almost_equal([food.position for food in base_abc_obj.foods],
                                  [[ 0.9762700785464951,  4.3037873274483900],
                                   [ 2.0552675214328780,  0.8976636599379368],
                                   [-1.5269040132219054,  2.9178822613331230],
                                   [-1.2482557747461502,  7.8354600156415940],
                                   [ 9.2732552100205870, -2.3311696234844455]], decimal=4)

# %%
def test_fit_solution():
    global base_abc_obj
    abc_obj = deepcopy(base_abc_obj)
    rng.seed(1)
    abc_obj.fit()
    # return abc_obj.get_solution()
    npt.assert_array_almost_equal(abc_obj.get_solution(),
                                  [-0.09354916012092745, -0.19970219343732298], decimal=4)

# %%
def test_get_agents():
    global base_abc_obj
    abc_obj = deepcopy(base_abc_obj)
    rng.seed(2)
    abc_obj.fit()
    # return abc_obj.get_agents()
    npt.assert_array_almost_equal(abc_obj.get_agents(),
                                  [[[ 0.97627007854649510,  4.303787327448390000],
                                    [ 2.05526752143287800,  0.897663659937936800],
                                    [-1.52690401322190540,  2.917882261333123000],
                                    [-1.24825577474615020,  7.835460015641594000],
                                    [ 9.27325521002058700, -2.331169623484445500]],
                                   [[ 0.90841802156529600,  4.303787327448390000],
                                    [ 2.05526752143287800,  0.897663659937936800],
                                    [-1.52690401322190540,  2.917882261333123000],
                                    [-1.24825577474615020,  7.835460015641594000],
                                    [ 9.27325521002058700, -0.165775937113451300]],
                                   [[ 0.22524265160989365,  4.303787327448390000],
                                    [ 1.79008214325850610,  0.897663659937936800],
                                    [-1.52690401322190540,  2.917882261333123000],
                                    [-0.85436725314137900,  7.835460015641594000],
                                    [ 9.27325521002058700, -0.165775937113451300]],
                                   [[ 0.22524265160989365,  4.303787327448390000],
                                    [ 1.79008214325850610,  0.063203989150250100],
                                    [-1.52690401322190540,  2.470286683777233300],
                                    [-0.85436725314137900,  7.835460015641594000],
                                    [ 6.87691675023305600, -0.165775937113451300]],
                                   [[ 0.22524265160989365,  4.303787327448390000],
                                    [-1.32223120680479830, -0.012316033603716914],
                                    [-1.52690401322190540, -0.572619127023161400],
                                    [-0.85436725314137900,  4.375861228107361000],
                                    [ 6.87691675023305600, -0.165775937113451300]],
                                   [[ 0.22524265160989365,  0.009977305085013377],
                                    [-1.32223120680479830, -0.012316033603716914],
                                    [-1.52690401322190540,  0.494478263254604000],
                                    [-0.85436725314137900,  4.375861228107361000],
                                    [ 6.87691675023305600, -0.165775937113451300]],
                                   [[ 7.34763074313047000,  5.846604386125158000],
                                    [-1.32223120680479830, -0.008502091679042274],
                                    [-1.52690401322190540,  0.494478263254604000],
                                    [-0.85436725314137900,  1.449558901725805700],
                                    [ 0.72494771745324550,  0.066032131534398440]],
                                   [[ 0.69560906834473530,  5.846604386125158000],
                                    [-1.04693413981496520, -0.008502091679042274],
                                    [-1.44480924566397450,  0.494478263254604000],
                                    [-0.85436725314137900,  1.325558203445703600],
                                    [ 0.72494771745324550,  0.066032131534398440]],
                                   [[ 0.44766762012532790,  5.846604386125158000],
                                    [-0.14895807508384940, -0.008502091679042274],
                                    [-1.35554547475358330,  0.494478263254604000],
                                    [-0.85436725314137900,  1.325558203445703600],
                                    [ 0.72494771745324550,  0.066032131534398440]],
                                   [[ 0.44766762012532790,  5.846604386125158000],
                                    [-0.14895807508384940, -0.008502091679042274],
                                    [-0.47189867566037100,  0.494478263254604000],
                                    [ 0.07529808467073218,  1.325558203445703600],
                                    [-3.46256474293496800,  7.305906863428706500]],
                                   [[ 0.44766762012532790,  4.628327424519170000],
                                    [-0.14458427398252194, -0.008502091679042274],
                                    [-0.18576143396033630, -0.006946174715418363],
                                    [ 0.07529808467073218,  0.541391252321319800],
                                    [-2.89630663033126720,  7.305906863428706500]]], decimal=4)

# %%
def test_get_status():
    global base_abc_obj
    abc_obj = deepcopy(base_abc_obj)
    rng.seed(3)
    abc_obj.fit()
    # return abc_obj.get_status()
    npt.assert_array_almost_equal(abc_obj.get_status(),
                                  (10, 1), decimal=0)

# %%
def test_bin_food_source_generation():
    global base_bin_abc_obj
    # return [food.position for food in base_bin_abc_obj.bin_abc_object.foods]
    npt.assert_array_almost_equal([food.position for food in base_bin_abc_obj.bin_abc_object.foods],
                                  [[ 5.834500761,  0.577898395,  1.360891221,  8.511932765],
                                   [ 5.563135018,  7.400242964,  9.572366844,  5.983171284],
                                   [-7.132934251,  8.893378340,  0.436966435, -1.706761200],
                                   [-9.624203991,  2.352709941,  2.241914454,  2.338679937],
                                   [ 3.952623918, -8.795490567,  3.335334308,  3.412757392]], decimal=4)

# %%
def test_bin_fit_solution():
    global base_bin_abc_obj
    bin_abc_obj = deepcopy(base_bin_abc_obj)
    rng.seed(1)
    bin_abc_obj.fit()
    # return bin_abc_obj.get_solution()
    npt.assert_array_almost_equal(bin_abc_obj.get_solution(),
                                  [False, True, True, True], decimal=0)

# %%
def test_bin_get_agents():
    global base_bin_abc_obj
    bin_abc_obj = deepcopy(base_bin_abc_obj)
    rng.seed(2)
    bin_abc_obj.fit()
    # return bin_abc_obj.get_agents()
    npt.assert_array_almost_equal(bin_abc_obj.get_agents(),
                                  [[[ 5.834500761,  0.577898395,  1.360891221,  8.511932765],
                                    [ 5.563135018,  7.400242964,  9.572366844,  5.983171284],
                                    [-7.132934251,  8.893378340,  0.436966435, -1.706761200],
                                    [-9.624203991,  2.352709941,  2.241914454,  2.338679937],
                                    [ 3.952623918, -8.795490567,  3.335334308,  3.412757392]],
                                   [[ 5.834500761,  0.577898395,  1.360891221,  8.511932765],
                                    [ 5.563135018,  7.400242964,  9.572366844,  5.983171284],
                                    [-7.132934251,  8.893378340,  0.436966435, -1.706761200],
                                    [-9.624203991,  2.352709941,  2.241914454,  2.338679937],
                                    [ 3.952623918, -8.795490567,  3.335334308,  3.412757392]],
                                   [[ 5.834500761, -2.119676881,  1.360891221,  8.511932765],
                                    [ 5.563135018,  7.400242964,  9.572366844,  5.983171284],
                                    [-7.132934251,  8.893378340,  0.436966435, -1.706761200],
                                    [-9.624203991,  2.352709941,  2.241914454,  2.338679937],
                                    [ 3.952623918, -8.795490567,  3.335334308,  3.412757392]],
                                   [[ 5.834500761, -2.119676881,  1.360891221,  8.511932765],
                                    [ 5.563135018,  7.400242964,  9.572366844,  5.983171284],
                                    [-7.132934251,  8.893378340,  0.436966435, -1.706761200],
                                    [-9.969185165, -9.174617621, -1.487215134,  9.287937996],
                                    [ 3.952623918, -8.795490567,  3.335334308,  3.412757392]],
                                   [[ 5.834500761, -2.119676881,  1.360891221,  8.511932765],
                                    [ 5.563135018,  7.400242964,  9.572366844, -0.884089599],
                                    [-8.040792082, -0.065449953, -8.606788191,  8.703071267],
                                    [-9.969185165, -9.174617621, -1.487215134,  9.287937996],
                                    [ 3.952623918, -8.795490567,  3.335334308,  3.412757392]],
                                   [[ 5.834500761, -2.119676881,  1.360891221,  8.511932765],
                                    [ 5.563135018,  7.400242964,  9.572366844, -0.884089599],
                                    [-8.040792082, -0.065449953, -8.606788191,  8.703071267],
                                    [-9.969185165, -9.174617621, -1.487215134,  9.287937996],
                                    [ 3.952623918, -8.795490567, -7.859163428,  3.412757392]],
                                   [[ 5.834500761, -2.119676881,  1.360891221,  8.511932765],
                                    [ 5.563135018, -1.591429895,  9.572366844, -0.884089599],
                                    [-8.040792082, -0.065449953, -8.606788191,  8.703071267],
                                    [-9.969185165, -9.174617621, -1.487215134,  9.287937996],
                                    [ 3.952623918, -8.795490567, -7.859163428,  3.412757392]],
                                   [[ 5.834500761, -2.119676881,  1.360891221,  8.511932765],
                                    [ 5.563135018, -1.591429895,  9.572366844, -0.884089599],
                                    [-8.040792082, -0.065449953, -8.606788191,  8.703071267],
                                    [-9.969185165, -9.174617621, -1.487215134,  9.287937996],
                                    [ 3.952623918, -8.795490567, -7.859163428,  3.412757392]],
                                   [[ 5.834500761, -2.119676881, -1.561465274,  8.511932765],
                                    [ 5.563135018, -1.591429895,  9.572366844, -0.884089599],
                                    [-8.040792082, -0.065449953, -8.606788191,  8.703071267],
                                    [-9.969185165, -9.174617621, -1.487215134,  9.287937996],
                                    [ 3.952623918, -8.795490567, -7.859163428,  3.412757392]],
                                   [[ 5.834500761, -2.119676881, -1.561465274,  8.511932765],
                                    [ 5.563135018, -1.591429895,  9.572366844, -0.884089599],
                                    [-8.040792082, -0.065449953, -8.606788191,  8.703071267],
                                    [-9.969185165, -9.174617621, -1.487215134,  9.287937996],
                                    [ 2.684775516,  6.609461023,  8.186189370, -5.625063904]],
                                   [[ 5.834500761, -2.119676881, -1.561465274,  8.511932765],
                                    [ 5.563135018, -1.591429895,  9.572366844, -0.884089599],
                                    [ 1.787838979,  6.806895895, -5.415141177, -4.430945631],
                                    [-9.969185165, -9.174617621, -1.487215134,  9.287937996],
                                    [ 2.684775516,  6.609461023,  8.186189370, -5.625063904]]], decimal=4)

# %%
def test_bin_get_status():
    global base_bin_abc_obj
    bin_abc_obj = deepcopy(base_bin_abc_obj)
    rng.seed(3)
    bin_abc_obj.fit()
    # return bin_abc_obj.get_status()
    npt.assert_array_almost_equal(bin_abc_obj.get_status(),
                                  (10, 5), decimal=0)
# %%