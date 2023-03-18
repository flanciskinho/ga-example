import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2.callbacks import MiddleCallbackFunc, MiddleCallbackConditionFunc, ActionConditions
from geneticalgorithm2.classes import MiddleCallbackData


def custom_callback(condition: MiddleCallbackConditionFunc) -> MiddleCallbackFunc:

    def log_function(data: MiddleCallbackData):

        cond = condition(data)
        if cond:
            generation = data.current_generation
            best = data.last_generation.variables[0]
            score = data.last_generation.scores[0]
            validation = function(best)
            print(f"{{'generation': '{generation}', "
                  f"'best': {{'gene': '{best.tolist()}', 'train': '{score}', 'validation': {validation} }} }}")

        return data, False

    return log_function


def function(x: np.ndarray) -> float:  # X as 1d-numpy array
    return np.sum(x**2) + x.mean() + x[0]*x[2]  # some float result


var_type = ('real', 'int', 'int')
var_bound = [[0.5, 1.5], [1, 100], [0, 5]]

algorithm_param = {'max_num_iteration': 10, 'population_size': 25}

model = ga(function=function, dimension=len(var_type), variable_type=var_type,
           variable_boundaries=var_bound, algorithm_parameters=algorithm_param)

model.run(no_plot=True, progress_bar_stream=None,
          middle_callbacks=[custom_callback(ActionConditions.EachGen(generation_step=1))])

convergence = model.report
print('\n\nConverge:')
for cnt in range(len(convergence)):
    print(f'{cnt}\t{convergence[cnt]}')

# solution = model.result
# print('\n\nSolution')
# print(solution)
