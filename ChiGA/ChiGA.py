import random
import time

import joblib
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

from Genetic_Algorithm import GA
from data.bank import bank_data
from data.census import census_data
from data.compas import compas_data
from data.credit import credit_data
from utils.config import census, credit, bank, compas

# global

global_inputs = set()
global_inputs_list = []
local_disc_inputs = set()
local_disc_inputs_list = []

# tot_inputs
tot_inputs = set()

"""
census: 9,1 for gender, age, 8 for race
credit: 9,13 for gender,age
bank:   1 for age
compas: 1, 2, 3 sex, age, race
"""

dataset = "credit"
data_config = {"census": census, "credit": credit, "bank": bank, "compas": compas}
config = data_config[dataset]  # replace
sensitive_param = 13
input_bounds = config.input_bounds
print(input_bounds)
print(config.feature_name)

classifier_name = '../unfair_models/{}/MLP_unfair1.pkl'.format(dataset)  # replace
print(classifier_name)
model = joblib.load(classifier_name)


class Global_Discovery(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, iteration, params, input_bounds, sensitive_param):
        s = self.stepsize
        samples = []
        random.seed(time.time())
        while len(samples) < iteration:
            x = np.zeros(params)
            for i in range(params):
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])
            x[sensitive_param - 1] = input_bounds[sensitive_param - 1][0]
            samples.append(x)
        return samples


def evaluate_local(inp):
    inp0 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = int(input_bounds[sensitive_param - 1][0])
    # inp0 = np.array(inp0)
    # print(tuple[inp0])
    # quit()
    # tot_inputs.add(tuple(inp0))
    inp0 = np.reshape(inp0, (1, -1))

    tot_inputs.add(tuple(map(tuple, inp0)))

    min_pre = 1
    max_pre = 0

    sum0 = 0
    sum1 = 0

    for val in range(config.input_bounds[sensitive_param - 1][0], config.input_bounds[sensitive_param - 1][1] + 1):

        inp1 = [int(i) for i in inp]
        inp1[sensitive_param - 1] = val

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out1 = model.predict(inp1)
        if out1 == 1:
            sum1 += 1
        else:
            sum0 += 1

        pre1 = model.predict_proba(inp1)

        if min_pre > pre1[0, 1]:
            min_pre = pre1[0, 1]

        if max_pre < pre1[0, 1]:
            max_pre = pre1[0, 1]


    # if (sum0 != 0) and (sum1 != 0) and (tuple(inp0) not in local_disc_inputs):
    if (sum0 != 0) and (sum1 != 0) and (tuple(map(tuple, inp0)) not in local_disc_inputs):
        local_disc_inputs.add(tuple(map(tuple, list(inp0))))
        local_disc_inputs_list.append(inp0.tolist()[0])


        return abs(max_pre - min_pre) * 5, 1
        # return 3, 1

    return abs(max_pre - min_pre), 0
    # return 1, 0


def xai_fair_testing(max_global, max_local):
    print(dataset, sensitive_param, max_global, max_local, classifier_name)
    data_config = {"census": census, "credit": credit, "bank": bank, 'compas': compas}
    config = data_config[dataset]
    feature_names = config.feature_name
    class_names = config.class_name
    sens_name = config.sens_name[sensitive_param]
    params = config.params

    data = {"census": census_data, "credit": credit_data, "bank": bank_data, "compas": compas_data}
    # prepare the testing data and model

    X, Y, input_shape, nb_classes = data[dataset]()

    # print(X[0:10])
    '''
        in Chi2, The inputs must be positive; 
    '''
    for b in range(len(input_bounds)):
        # print(input_bounds[b])
        # print(input_bounds[b][0])
        if input_bounds[b][0] < 0:
            for x in X:
                x[b] -= input_bounds[b][0]
    '''
        The Output should be 1-dimension
    '''
    T = []
    for y in Y:
        T.append(y[1])
    Y = np.array(T)

    RANDOM_STATE = 1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)

    para_prob = chi2(X_train, y_train)

    para_prob = np.array(para_prob[0])


    percent_positive = para_prob / np.sum(para_prob)

    percent_negative = (1 / percent_positive)
    percent_negative = percent_negative / np.sum(percent_negative)

    print("prob_posi", percent_positive)
    print("prob_nega", percent_negative)


    start = time.time()

    model_name = classifier_name.split("/")[-1].split("_")[0]

    file_name = "chiga_raw_rev_slc_t51_c9_NC_NR_{}_{}{}_{}_1H.txt".format(model_name, dataset, sensitive_param,
                                                                          int(max_global / 100))


    f = open(file_name, "a")
    f.write("iter:" + str(iter) + "---------------------chiga_rev_NC_NR---------------------" + "\n" + "\n")
    f.write("max_global : {}-------------max_local : {}---------\n\n".format(max_global, max_local))
    f.close()

    global_discovery = Global_Discovery()

    train_samples = global_discovery(max_global, params, input_bounds, sensitive_param)
    train_samples = np.array(train_samples)

    np.random.shuffle(train_samples)

    print(train_samples.shape)

    seed = train_samples



    print("Randomly Generate {} Samples".format(max_global))


    for inp in seed:
        inp0 = [int(i) for i in inp]
        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))
        tot_inputs.add(tuple(map(tuple, inp0)))
        global_inputs.add(tuple(map(tuple, inp0)))
        global_inputs_list.append(inp0.tolist()[0])

    end = time.time()

    print('Total time:' + str(end - start))

    print("")
    print("Starting Local Search")

    '''
    ------------------------------------
              Genetic Algorithm
    ------------------------------------
    '''

    nums = global_inputs_list
    DNA_SIZE = len(input_bounds)

    cross_rate = 0.9
    mutation = 0.05
    iteration = max_local
    ga = GA(nums=nums, bound=input_bounds, func=evaluate_local, sensitive_param=sensitive_param, datasetName=dataset,
            DNA_SIZE=DNA_SIZE, cross_rate=cross_rate,
            mutation=mutation, mutation_positive=percent_positive, mutation_negative=percent_negative
            )
    # for random

    '''
        300 s 
    '''
    count = 300

    SAVEFILENAME = '../results/' + dataset + '/' + str(
        sensitive_param) + '/disc_samples_{}_raw_rev_slc_t51_c9_NC_NR_{}_1H.npy'.format(
        model_name,
        int(max_global))


    for i in range(iteration):
        ga.evolution()
        end = time.time()
        use_time = end - start
        if use_time >= count:
            f = open(file_name, "a")

            f.write("Percentage discriminatory inputs - " + str(
                float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100) + "\n")
            f.write("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)) + "\n")
            f.write("Total Inputs are " + str(len(tot_inputs)) + "\n")
            f.write('use time:' + str(end - start) + "\n" + "\n")
            f.close()

            print("Total Inputs are " + str(len(tot_inputs)))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
            print("Percentage discriminatory inputs - " + str(float(len(local_disc_inputs_list))
                                                              / float(len(tot_inputs)) * 100))

            print('use time:' + str(end - start))
            count += 300

        if i % 300 == 0:
            print("Epochs {}".format(i))
            print("Total Inputs are " + str(len(tot_inputs)))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
            print("Percentage discriminatory inputs - " + str(float(len(local_disc_inputs_list))
                                                              / float(len(tot_inputs)) * 100))

            print('use time:' + str(end - start))

        if use_time >= 3600:
            f = open(file_name, "a")
            f.write("------------------FINISH---------------------")
            f.write("Percentage discriminatory inputs - " + str(
                float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100) + "\n")
            f.write("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)) + "\n")
            f.write("Total Inputs are " + str(len(tot_inputs)) + "\n")
            f.write('use time:' + str(end - start) + "\n" + "\n")
            f.write('saved to :' + SAVEFILENAME)
            f.close()

            print("---------------------FINISH----------------------")
            print("Epochs {}".format(i))
            print("Total Inputs are " + str(len(tot_inputs)))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
            print("Percentage discriminatory inputs - " + str(float(len(local_disc_inputs_list))
                                                              / float(len(tot_inputs)) * 100))
            print(dataset, sensitive_param, model_name)
            print('use time:' + str(end - start))
            print('saved to' + SAVEFILENAME)

            np.save(SAVEFILENAME, np.array(local_disc_inputs_list))

            # print("Time :", use_time)

            return

    # np.save(
    #     '../results/' + dataset + '/' + str(sensitive_param) + '/local_samples_{}_chi_rev_{}_{}.npy'.format(
    #         model_name,
    #         int(max_global / 100),
    #         int(max_local / 100)),
    #     np.array(local_disc_inputs_list))
    #
    # print("Total Inputs are " + str(len(tot_inputs)))
    # print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
    # print("Percentage discriminatory inputs - " + str(
    #     float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
    # print(
    #     "saved as " + '../results/' + dataset + '/' + str(
    #         sensitive_param) + '/local_samples_{}_chi_rev_{}_{}.npy'.format(
    #         model_name,
    #         int(max_global / 100),
    #         int(max_local / 100)))


def main(argv=None):
    xai_fair_testing(max_global=1000, max_local=500000)


if __name__ == '__main__':
    main()
