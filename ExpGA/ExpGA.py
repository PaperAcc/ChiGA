import joblib
import numpy as np
import random
import time
import utils.config
import sklearn
from sklearn.feature_selection import chi2
import lime
from lime.lime_tabular import LimeTabularExplainer
from data.bank import bank_data
from data.credit import credit_data
from data.census import census_data
from data.compas import compas_data
from Genetic_Algorithm import GA
from utils.config import bank, census, credit, compas

# from keras.models import load_model
# import signal
# import keras.backend as K
# from scipy.spatial.distance import cdist
# import copy

# global 记录的是 可能歧视的 输入 集
global_disc_inputs = set()
global_disc_inputs_list = []
local_disc_inputs = set()
local_disc_inputs_list = []

# tot_inputs 记录的是 所有产生的 输入 集
tot_inputs = set()
# 记录的是 sens 出现的 index ++
location = np.zeros(21)

# threshold_l = 7  # replace census-7,credit-14,bank-10
"""
census: 9,1 for gender, age, 8 for race
credit: 9,13 for gender,age
bank:   1 for age
compas: 1, 2, 3 sex, age, race
"""

dataset = "compas"
sensitive_param = 3

data_config = {"census": census, "credit": credit, "bank": bank, "compas": compas}
threshold_config = {"census": 7, "credit": 14, "bank": 10, "compas": 7}
threshold_l = threshold_config[dataset]  # replace census-7,credit-14,bank-10
config = data_config[dataset]  # replace
threshold = 0
input_bounds = config.input_bounds
print(dataset)
print(input_bounds)
print(config.feature_name)
classifier_name = '../ExpGA_fair_models/{}/{}/MLP_fair1.pkl'.format(dataset, sensitive_param)  # replace
model = joblib.load(classifier_name)
print(model)


def ConstructExplainer(train_vectors, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(train_vectors, feature_names=feature_names,
                                                       class_names=class_names, discretize_continuous=False)
    return explainer


'''
    LIME 局部解释器 （已经排序）
    选取 rank 靠前的
'''


def Searchseed(model, feature_names, sens_name, explainer, train_vectors, num, X_ori):
    seed = []
    # temp_time = time.time()
    for x in train_vectors:
        tot_inputs.add(tuple(x))
        exp = explainer.explain_instance(x, model.predict_proba, num_features=num)

        # print(time.time() - temp_time, "s", exp)

        explain_labels = exp.available_labels()
        exp_result = exp.as_list(label=explain_labels[0])
        rank = []
        for j in range(len(exp_result)):
            rank.append(exp_result[j][0])
        loc = rank.index(sens_name)
        location[loc] = location[loc] + 1
        if loc < threshold_l:
            seed.append(x)

            '''
            imp 没有使用？
            '''

            # imp = []
            # for item in feature_names:
            #     pos = rank.index(item)
            #     imp.append(exp_result[pos][1])
        if len(seed) >= 200:
            return seed
    return seed


class Global_Discovery(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, iteration, params, input_bounds, sensitive_param):
        s = self.stepsize
        random.seed(time.time())
        samples = []
        while len(samples) < iteration:
            x = np.zeros(params)
            for i in range(params):
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])
            x[sensitive_param - 1] = config.input_bounds[sensitive_param - 1][0]
            samples.append(x)
        return samples


def evaluate_local(inp):
    inp_ = [int(i) for i in inp]
    inp_[sensitive_param - 1] = int(config.input_bounds[sensitive_param - 1][0])
    # print(inp_)
    tot_inputs.add(tuple(inp_))
    pre0 = 0
    # quit()
    pre1 = 0
    for val in range(config.input_bounds[sensitive_param - 1][0], config.input_bounds[sensitive_param - 1][1] + 1):
        if val != inp_[sensitive_param - 1]:
            # 进行一个替换
            inp1 = [int(i) for i in inp_]
            inp1[sensitive_param - 1] = val

            inp0 = np.asarray(inp_)
            inp0 = np.reshape(inp0, (1, -1))

            inp1 = np.asarray(inp1)
            inp1 = np.reshape(inp1, (1, -1))

            out0 = model.predict(inp0)
            out1 = model.predict(inp1)

            # pre0 = model.predict_proba(inp0)[0]
            # pre1 = model.predict_proba(inp1)[0]

            # print(abs(pre0 - pre1)[0])
            # quit()
            # print((tuple(map(tuple, inp0))))
            if (abs(out0 - out1) > threshold and (tuple(map(tuple, inp0)) not in global_disc_inputs)
                    and (tuple(map(tuple, inp0)) not in local_disc_inputs)):
                # print(tuple(map(tuple, inp0)))

                local_disc_inputs.add(tuple(map(tuple, list(inp0))))
                local_disc_inputs_list.append(inp0.tolist()[0])
                # print(pre0, pre1)
                # print(out1, out0)

                return 2 * abs(out1 - out0) + 1
                # return abs(pre0 - pre1)
    # return abs(pre0 - pre1)
    return 2 * abs(out1 - out0) + 1

    # return not abs(out0 - out1) > threshold
    # for binary classification, we have found that the
    # following optimization function gives better results


def xai_fair_testing(max_global, max_local):
    # print(dataset)
    # print(input_bounds)
    # print(config.feature_name)
    # print(model)
    # quit()

    start = time.time()
    print(dataset, sensitive_param, max_global, max_local, classifier_name)

    data_config = {"census": census, "credit": credit, "bank": bank, "compas": compas}
    config = data_config[dataset]
    feature_names = config.feature_name
    class_names = config.class_name
    sens_name = config.sens_name[sensitive_param]
    params = config.params

    print(sens_name)
    print(class_names)
    print(params)

    data = {"census": census_data, "credit": credit_data, "bank": bank_data, "compas": compas_data}
    # prepare the testing data and unfair_models
    X, Y, input_shape, nb_classes = data[dataset]()

    start = time.time()

    model_name = classifier_name.split("/")[-1].split("_")[0]
    # file_name = "aequitas_"+dataset+sensitive_param+"_"+unfair_models+""
    # file_name = "expga_{}_{}{}.txt".format(model_name, dataset, sensitive_param)
    file_name = "expga_ExpGA_retrain_{}_{}{}_{}_{}_1H.txt".format(model_name, dataset, sensitive_param,
                                                                  int(max_global / 100),
                                                                  int(max_local / 100))
    f = open(file_name, "a")
    f.write("iter:" + str(iter) + "------------------------------------------" + "\n" + "\n")
    f.close()

    global_discovery = Global_Discovery()
    '''
    train_samples 随机输入集
    与训练数据无关
    '''
    train_samples = global_discovery(max_global, params, input_bounds, sensitive_param)
    train_samples = np.array(train_samples)
    # train_samples = X[np.random.choice(X.shape[0], max_global, replace=False)]

    np.random.shuffle(train_samples)

    print(train_samples.shape)
    explainer = ConstructExplainer(X, feature_names, class_names)

    '''
        没有使用 X 输入数据集
        返回的是 可能歧视数据集
    '''

    print("-------Explainer Constructed!------")
    print("-------Searching Seed--------------")
    seed = Searchseed(model, feature_names, sens_name, explainer, train_samples, params, X)

    print('Finish Searchseed')

    for inp in seed:
        inp0 = [int(i) for i in inp]
        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))
        global_disc_inputs.add(tuple(map(tuple, inp0)))
        global_disc_inputs_list.append(inp0.tolist()[0])

    print("Finished Global Search")
    # total
    print('length of total input is:' + str(len(tot_inputs)))
    # potential
    print('length of global discovery is:' + str(len(global_disc_inputs_list)))

    end = time.time()

    print('Total time:' + str(end - start))

    print("")
    print("Starting Local Search")

    '''
    ------------------------------------
              Genetic Algorithm
    ------------------------------------
    '''

    nums = global_disc_inputs_list
    DNA_SIZE = len(input_bounds)
    cross_rate = 0.9
    mutation = 0.05
    iteration = max_local
    ga = GA(nums=nums, bound=input_bounds, func=evaluate_local, DNA_SIZE=DNA_SIZE, cross_rate=cross_rate,
            mutation=mutation)
    # for random

    '''
        每 300 s 打印一次
    '''
    count = 300
    for i in range(iteration):
        ga.evolution()
        end = time.time()
        use_time = end - start
        if use_time >= count:
            f = open(file_name, "a")
            f.write("Percentage discriminatory inputs - " + str(
                float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                / float(len(tot_inputs)) * 100) + "\n")
            f.write("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)) + "\n")
            f.write("Total Inputs are " + str(len(tot_inputs)) + "\n")
            f.write('use time:' + str(end - start) + "\n" + "\n")
            f.close()

            print("Percentage discriminatory inputs - " + str(float(len(local_disc_inputs_list))
                                                              / float(len(tot_inputs)) * 100))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
            print("Total Inputs are " + str(len(tot_inputs)) + "\n")

            print('use time:' + str(end - start))
            print("Epoch ", i, " / ", iteration)
            count += 300

        if i % 300 == 0:
            print("Epochs :", i)
            print('use time:' + str(end - start))
            print("Total Inputs are " + str(len(tot_inputs)))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
            print("Percentage discriminatory inputs - " + str(
                float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))

        if use_time >= 3600:
            print("---------------FINISH-----------------")
            f = open(file_name, "a")
            f.write("-------------FINISH------------------")
            f.write("Percentage discriminatory inputs - " + str(
                float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                / float(len(tot_inputs)) * 100) + "\n")
            f.write("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)) + "\n")
            f.write("Total Inputs are " + str(len(tot_inputs)) + "\n")
            f.write('use time:' + str(end - start) + "\n" + "\n")
            f.write("-------------FINISH------------------")
            f.close()
            print("Epochs :", i)
            print('use time:' + str(end - start))
            print("Total Inputs are " + str(len(tot_inputs)))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
            print("Percentage discriminatory inputs - " + str(
                float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
            print(dataset, sensitive_param, model_name)
            print("---------------FINISH-----------------")
            np.save(
                '../ExpGA_results/' + dataset + '/{}/local_samples_{}_expga_{}_1H.npy'.format(sensitive_param, model_name,
                                                                                        max_global),
                np.array(local_disc_inputs_list))

            return

    np.save(
        '../results/' + dataset + '/{}/local_samples_{}_expga_{}.npy'.format(sensitive_param, model_name, max_global),
        np.array(local_disc_inputs_list))

    # print("Total Inputs are " + str(len(tot_inputs)))
    # print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))


def main(argv=None):
    xai_fair_testing(max_global=1000, max_local=500000)


if __name__ == '__main__':
    main()
