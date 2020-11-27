# %%
from beecolpy import *

from copy import deepcopy
import math as mt
import numpy as np
import random as rng
import numpy.testing as npt

def sphere(x): #Continuous and NaN benchmark
    #Sphere function
    #Min: x=(0,0)
    total = 0
    test = 0
    for i in range(len(x)):
        total += x[i]**2
        test += x[i]
    if (test<5):
        return total
    else:
        return np.nan

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

base_bin_abc_obj = bin_abc(squared_bin, bits_count=4,
                          transfer_function='sigmoid',
                          colony_size=10, scouts=0.5,
                          iterations=10, min_max='min',
                          method='bin',
                          nan_protection=3)

base_am_abc_obj = bin_abc(squared_bin, bits_count=4,
                        colony_size=10, scouts=0.5,
                        iterations=10, min_max='min',
                        method='am',
                        nan_protection=True)

# %%
def test_food_source_generation():
    # Test algorithm initialization
    global base_abc_obj
    # return [food.position for food in base_abc_obj.foods]
    npt.assert_array_almost_equal([food.position for food in base_abc_obj.foods],
                                  [[-1.5885683833831, -4.821664994140733],
                                   [ 0.2254944273721, -1.901317250991713],
                                   [ 5.6759717806954, -3.933745478421451],
                                   [-0.4680609169528,  1.667640789100623],
                                   [-4.3632431120059,  5.116084083144479]],
                                   decimal=6)

# %%
def test_fit_solution():
    # Test solver capability
    global base_abc_obj
    abc_obj = deepcopy(base_abc_obj)
    rng.seed(1)
    abc_obj.fit()
    # return abc_obj.get_solution()
    npt.assert_array_almost_equal(abc_obj.get_solution(),
                                  [ 0.04921418749376193,  0.5454139496754725],
                                  decimal=6)

# %%
def test_get_agents():
    # Verifies the process step-by-step
    global base_abc_obj
    abc_obj = deepcopy(base_abc_obj)
    rng.seed(2)
    abc_obj.fit()
    # return abc_obj.get_agents()
    npt.assert_array_almost_equal(abc_obj.get_agents(),
                                  [[[-1.5885683833831, -4.82166499414073],
                                    [ 0.2254944273721, -1.90131725099171],
                                    [ 5.6759717806954, -3.93374547842145],
                                    [-0.4680609169528,  1.66764078910062],
                                    [-4.3632431120059,  5.11608408314447]],
                                   [[-1.5885683833831, -4.82166499414073],
                                    [ 0.2254944273721,  0.65667238615170],
                                    [ 5.6759717806954, -3.93374547842145],
                                    [ 0.0033368448204, -0.33071580563951],
                                    [-4.3632431120059,  5.11608408314447]],
                                   [[-1.5885683833831, -4.82166499414073],
                                    [ 0.2254944273721,  0.65667238615170],
                                    [ 5.6759717806954, -3.93374547842145],
                                    [ 0.0033368448204,  0.10550786822243],
                                    [-4.3632431120059,  5.11608408314447]],
                                   [[ 0.3755850514341, -4.16067485896215],
                                    [ 0.2254944273721,  0.65667238615170],
                                    [ 0.5450264275880, -3.93374547842145],
                                    [-0.5444097580337, -5.48944859304224],
                                    [-4.3632431120059,  5.11608408314447]],
                                   [[ 0.3615935787167, -4.16067485896215],
                                    [-6.3647984339914, -5.36949141245718],
                                    [ 0.3746170951623, -1.72204931373615],
                                    [-0.5444097580337, -5.48944859304224],
                                    [-0.1393402662189,  5.11608408314447]],
                                   [[ 0.3615935787167, -4.16067485896215],
                                    [-6.3647984339914, -5.36949141245718],
                                    [-0.3513837598430, -1.72204931373615],
                                    [-0.5444097580337, -4.31796153278332],
                                    [-0.1393402662189,  5.11608408314447]],
                                   [[-0.0388640057100, -4.16067485896215],
                                    [-1.8920435755067, -5.36949141245718],
                                    [-0.1531056064629, -1.72204931373615],
                                    [-0.5444097580337, -4.31796153278332],
                                    [-0.1393402662189,  5.11608408314447]],
                                   [[-0.0388640057100, -4.16067485896215],
                                    [-1.8920435755067, -5.36949141245718],
                                    [-0.1531056064629, -1.72204931373615],
                                    [-0.5444097580337, -3.37708223915825],
                                    [-0.1393402662189,  0.92206396689625]],
                                   [[-0.0388640057100, -4.16067485896215],
                                    [-1.8920435755067, -5.36949141245718],
                                    [-0.1531056064629, -1.72204931373615],
                                    [-0.5444097580337, -2.11672284255737],
                                    [-0.1281716938908,  0.73965692056622]],
                                   [[ 0.0066238987307, -4.16067485896215],
                                    [-1.8920435755067, -5.10475297516781],
                                    [-0.1464851678880,  0.60412612558928],
                                    [-0.5444097580337, -2.11672284255737],
                                    [-0.1281716938908,  0.73965692056622]],
                                   [[ 0.0066238987307, -3.97904878827678],
                                    [-1.2149964534247, -5.10475297516781],
                                    [-0.1464851678880,  0.21345813510979],
                                    [-0.4298854019016,  0.38963726112005],
                                    [-0.1281716938908,  0.73965692056622]]],
                                    decimal=6)

# %%
def test_get_status():
    # Test exploration and NaN protection
    global base_abc_obj
    abc_obj = deepcopy(base_abc_obj)
    rng.seed(3)
    abc_obj.fit()
    # return abc_obj.get_status()
    npt.assert_array_almost_equal(abc_obj.get_status(),
                                  (10, 1, 2), decimal=0)

# %%
def test_bin_food_source_generation():
    # Test algorithm initialization
    global base_bin_abc_obj
    # return [food.position for food in base_bin_abc_obj._bin_abc_object.foods]
    npt.assert_array_almost_equal([food.position for food in base_bin_abc_obj._bin_abc_object.foods],
                                  [[ 2.36737993350663, -4.9898731727511,  8.194925119364804,  9.655709520753064],
                                   [ 7.97676575935987,  3.6796786383088, -0.557145690945732, -7.985975838632684],
                                   [-0.45980446894565,  7.3061985554328, -4.790153792160812,  6.100556540260445],
                                   [ 6.49689954296465,  3.3630640246370, -9.977143613711434, -0.128442670693507],
                                   [-6.17865816995219,  1.3502148124134, -5.227681427695597,  9.350805005802869]],
                                   decimal=6)

# %%
def test_bin_fit_solution():
    # Test solver capability
    global base_bin_abc_obj
    bin_abc_obj = deepcopy(base_bin_abc_obj)
    rng.seed(1)
    bin_abc_obj.fit()
    # return bin_abc_obj.get_solution()
    npt.assert_array_almost_equal(bin_abc_obj.get_solution(),
                                  [True, False, False, False],
                                  decimal=0)

# %%
def test_bin_get_agents():
    # Verifies the process step-by-step
    global base_bin_abc_obj
    bin_abc_obj = deepcopy(base_bin_abc_obj)
    rng.seed(2)
    bin_abc_obj.fit()
    # return bin_abc_obj.get_agents()
    npt.assert_array_almost_equal(bin_abc_obj.get_agents(),
                                  [[[ 2.36737993350663, -4.9898731727511,  8.1949251193648,  9.65570952075306],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [-0.45980446894565,  7.3061985554328, -4.7901537921608,  6.10055654026044],
                                    [ 6.49689954296465,  3.3630640246370, -9.9771436137114, -0.12844267069350],
                                    [-6.17865816995219,  1.3502148124134, -5.2276814276955,  9.35080500580286]],
                                   [[ 2.36737993350663, -4.9898731727511,  8.1949251193648,  9.65570952075306],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [-0.45980446894565,  7.3061985554328, -4.7901537921608,  6.10055654026044],
                                    [ 6.49689954296465,  3.3630640246370, -9.9771436137114, -0.12844267069350],
                                    [-6.17865816995219,  1.3502148124134, -5.2276814276955,  9.35080500580286]],
                                   [[ 2.36737993350663, -4.9898731727511,  8.1949251193648,  9.65570952075306],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [-0.45980446894565,  7.3061985554328, -4.7901537921608,  6.10055654026044],
                                    [ 6.83559391534327,  3.3630640246370, -9.9771436137114, -0.12844267069350],
                                    [-6.17865816995219,  1.3502148124134, -5.2276814276955,  9.35080500580286]],
                                   [[ 2.36737993350663, -4.9898731727511,  8.1949251193648,  9.65570952075306],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [-0.45980446894565,  7.3061985554328, -4.7901537921608,  6.10055654026044],
                                    [ 6.83559391534327,  3.3630640246370, -9.9771436137114, -0.12844267069350],
                                    [-6.17865816995219,  1.3502148124134, -5.2276814276955,  9.35080500580286]],
                                   [[ 2.36737993350663, -4.9898731727511,  8.1949251193648,  9.65570952075306],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [ 7.69359454104137, -7.9908523872518,  6.3124243256037,  5.34001481556772],
                                    [ 6.83559391534327,  3.3630640246370, -9.9771436137114, -0.12844267069350],
                                    [-6.17865816995219,  1.3502148124134, -5.2276814276955,  9.35080500580286]],
                                   [[ 2.36737993350663, -4.9898731727511,  8.1949251193648,  9.65570952075306],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [ 7.69359454104137, -7.9908523872518,  6.3124243256037,  5.34001481556772],
                                    [ 6.83559391534327,  3.3630640246370, -9.9771436137114, -0.12844267069350],
                                    [-6.20907593483815,  3.5694131025996, -2.5242334328093, -2.87804169353179]],
                                   [[ 2.36737993350663, -4.9898731727511,  0.2262684675870,  9.65570952075306],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [ 7.69359454104137, -7.9908523872518,  6.3124243256037,  5.34001481556772],
                                    [ 7.57702680055116, -0.9733738012227,  1.8267864179924, -7.63094783762329],
                                    [-6.20907593483815,  3.5694131025996, -2.5242334328093,  9.44549387949469]],
                                   [[ 2.36737993350663, -4.9898731727511,  0.2262684675870,  9.65570952075306],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [ 7.69359454104137, -7.9908523872518,  6.3124243256037,  5.34001481556772],
                                    [ 7.57702680055116, -0.9733738012227,  1.8267864179924, -7.63094783762329],
                                    [-6.20907593483815,  3.5694131025996, -2.5242334328093,  9.44549387949469]],
                                   [[ 2.36737993350663, -4.9898731727511,  0.2262684675870,  9.65570952075306],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [ 7.69359454104137, -7.9908523872518,  6.3124243256037,  5.34001481556772],
                                    [ 7.57702680055116, -0.9733738012227,  1.8267864179924, -7.63094783762329],
                                    [-6.20907593483815,  3.5694131025996, -2.5242334328093,  9.44549387949469]],
                                   [[ 7.24313015673519, -8.4654330178499,  0.8063731352420,  2.19236117155752],
                                    [ 7.97676575935987,  3.6796786383088, -0.5571456909457, -7.98597583863268],
                                    [ 7.69359454104137, -7.9908523872518,  6.3124243256037,  5.34001481556772],
                                    [ 7.57702680055116, -4.6139155958262,  1.8267864179924, -7.63094783762329],
                                    [-6.20907593483815,  3.5694131025996, -2.5242334328093,  9.44549387949469]],
                                   [[ 7.24313015673519, -8.4654330178499,  0.8063731352420,  2.19236117155752],
                                    [-8.19692150306741,  4.8381040941882,  6.6605368838552, -0.94906961675102],
                                    [ 7.69359454104137, -7.9908523872518,  6.3124243256037,  5.34001481556772],
                                    [ 7.57702680055116, -4.6139155958262,  1.8267864179924, -7.63094783762329],
                                    [-6.20907593483815,  3.5694131025996, -2.5242334328093,  9.44549387949469]]],
                                    decimal=6)

# %%
def test_bin_get_status():
    # Test exploration
    global base_bin_abc_obj
    bin_abc_obj = deepcopy(base_bin_abc_obj)
    rng.seed(3)
    bin_abc_obj.fit()
    # return bin_abc_obj.get_status()
    npt.assert_array_almost_equal(bin_abc_obj.get_status(),
                                  (10, 5, 0), decimal=0)

# %%
def test_am_food_source_generation():
    # Test algorithm initialization
    global base_am_abc_obj
    # return [food.position for food in base_am_abc_obj._bin_abc_object.foods]
    npt.assert_array_almost_equal([food.position for food in base_am_abc_obj._bin_abc_object.foods],
                                  [[ 0.0317625700822956,  1.731335296907627, -1.563768616275585,  0.205068984362204],
                                   [ 0.8262456394675586,  0.189763645313695,  1.257867453165344,  0.161134427881295],
                                   [ 1.8553541838952037,  0.412742511845531,  0.350468256701745, -0.220043894897935],
                                   [ 0.3851474463324251, -0.460395416109358,  0.302604056659554, -0.838681990388968],
                                   [-1.2424346857825754, -1.253081886977779,  0.451092719474426,  0.626637555958515]],
                                   decimal=6)

# %%
def test_am_fit_solution():
    # Test solver capability
    global base_am_abc_obj
    am_abc_obj = deepcopy(base_am_abc_obj)
    rng.seed(1)
    am_abc_obj.fit()
    # return am_abc_obj.get_solution()
    npt.assert_array_almost_equal(am_abc_obj.get_solution(),
                                  [True, False, False, False], decimal=0)

# %%
def test_am_get_agents():
    # Verifies the process step-by-step
    global base_am_abc_obj
    am_abc_obj = deepcopy(base_am_abc_obj)
    rng.seed(2)
    am_abc_obj.fit()
    # return am_abc_obj.get_agents()
    npt.assert_array_almost_equal(am_abc_obj.get_agents(),
                                  [[[ 0.0317625700822956,  1.731335296907627, -1.563768616275585,  0.20506898436220],
                                    [ 0.8262456394675586,  0.189763645313695,  1.257867453165344,  0.16113442788129],
                                    [ 1.8553541838952037,  0.412742511845531,  0.350468256701745, -0.22004389489793],
                                    [ 0.3851474463324251, -0.460395416109358,  0.302604056659554, -0.83868199038896],
                                    [-1.2424346857825754, -1.253081886977779,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -1.563768616275585,  0.20506898436220],
                                    [ 0.8262456394675586,  0.351781252353573,  1.257867453165344,  0.16113442788129],
                                    [ 1.8553541838952037,  0.412742511845531,  0.350468256701745, -0.22004389489793],
                                    [ 0.3851474463324251, -0.460395416109358,  0.216555578178622, -1.11557382498281],
                                    [-1.2424346857825754, -1.253081886977779,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -1.563768616275585,  0.20506898436220],
                                    [ 0.8262456394675586,  0.351781252353573,  1.257867453165344,  0.16113442788129],
                                    [ 1.8553541838952037,  0.412742511845531,  0.350468256701745, -0.22004389489793],
                                    [ 0.3851474463324251, -0.460395416109358,  0.216555578178622, -1.11557382498281],
                                    [-1.2424346857825754, -1.253081886977779,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -1.563768616275585,  0.20506898436220],
                                    [ 0.8262456394675586,  0.351781252353573,  1.257867453165344,  0.16113442788129],
                                    [ 1.8553541838952037,  0.412742511845531,  0.350468256701745, -0.22004389489793],
                                    [ 0.3851474463324251, -0.460395416109358,  0.216555578178622, -1.11557382498281],
                                    [-1.2424346857825754, -1.253081886977779,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -1.563768616275585,  0.20506898436220],
                                    [ 0.8262456394675586,  0.351781252353573,  1.257867453165344,  0.16113442788129],
                                    [-1.7058093473082554, -1.678283137344201, -1.087187122578500,  1.16633432508133],
                                    [ 0.3851474463324251, -0.460395416109358,  0.216555578178622, -1.11557382498281],
                                    [-1.2424346857825754, -1.253081886977779,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -0.582971548063298,  0.20506898436220],
                                    [ 0.8262456394675586,  0.351781252353573,  1.257867453165344,  0.16113442788129],
                                    [-1.7058093473082554, -1.678283137344201, -1.087187122578500,  1.16633432508133],
                                    [ 0.3851474463324251, -0.460395416109358,  0.216555578178622, -1.11557382498281],
                                    [-1.2424346857825754,  1.123043984719443,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -0.582971548063298,  0.20506898436220],
                                    [-1.6779571289854807,  1.400283495619391,  0.563966375324260,  1.83867425358324],
                                    [-1.7058093473082554, -1.678283137344201, -1.087187122578500,  1.16633432508133],
                                    [ 0.3851474463324251, -0.460395416109358,  0.216555578178622, -1.11557382498281],
                                    [-1.2424346857825754,  1.123043984719443,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -0.582971548063298, -0.02865869427732],
                                    [-1.6779571289854807,  1.400283495619391,  0.465013092364492,  1.83867425358324],
                                    [-1.7058093473082554, -1.678283137344201, -1.087187122578500,  1.16633432508133],
                                    [-1.0732014266132368,  0.682929303799766,  0.724396029991348, -0.24461665235368],
                                    [-1.2424346857825754,  1.123043984719443,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -0.582971548063298, -0.02865869427732],
                                    [-1.6779571289854807,  1.400283495619391,  0.465013092364492,  1.83867425358324],
                                    [-1.7058093473082554, -1.678283137344201, -1.087187122578500,  1.16633432508133],
                                    [-1.0732014266132368,  0.682929303799766,  0.724396029991348, -0.24461665235368],
                                    [-1.1791190928582518,  1.123043984719443,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -0.582971548063298, -0.02865869427732],
                                    [-1.6779571289854807,  1.400283495619391,  0.465013092364492,  1.83867425358324],
                                    [-1.7058093473082554, -1.678283137344201, -1.087187122578500,  1.16633432508133],
                                    [-1.0732014266132368,  0.682929303799766,  0.724396029991348, -0.24461665235368],
                                    [-1.1791190928582518,  1.123043984719443,  0.451092719474426,  0.62663755595851]],
                                   [[ 0.0317625700822956,  1.731335296907627, -0.582971548063298, -0.02865869427732],
                                    [-1.6779571289854807,  1.400283495619391,  0.465013092364492,  1.83867425358324],
                                    [-1.7058093473082554, -1.678283137344201, -0.758135673544020,  1.16633432508133],
                                    [-1.0732014266132368,  0.682929303799766,  0.724396029991348, -0.24461665235368],
                                    [-1.1791190928582518,  1.123043984719443,  0.451092719474426,  0.62663755595851]]],
                                    decimal=6)

# %%
def test_am_get_status():
    # Test exploration
    global base_am_abc_obj
    am_abc_obj = deepcopy(base_am_abc_obj)
    rng.seed(3)
    am_abc_obj.fit()
    # return am_abc_obj.get_status()
    npt.assert_array_almost_equal(am_abc_obj.get_status(),
                                  (10, 4, 0), decimal=0)

# %%
