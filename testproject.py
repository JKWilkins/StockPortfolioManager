import numpy as np
import ManualStrategy as ms
import experiment1 as exp1
import experiment2 as exp2



def author():
    return 'jwilkins36'


if __name__ == "__main__":
    np.random.seed(2022)

    # ManualStrategy
    ms.msplots()

    # Experiment 1
    exp1.experiment1()

    # Experiment 2
    exp2.experiment2()

