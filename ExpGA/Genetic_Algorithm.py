import numpy as np


# import pandas as pd


# classifier_name = 'Random_Forest_standard_unfair.pkl'
# unfair_models = joblib.load(classifier_name)

class GA:
    # input:
    #     nums: m * n  n is nums_of x, y, z, ...,and m is population's quantity
    #     bound:n * 2  [(min, nax), (min, max), (min, max),...]
    #     DNA_SIZE is binary bit size, None is auto
    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
        nums = np.array(nums)
        bound = np.array(bound)
        self.bound = bound
        min_nums, max_nums = np.array(list(zip(*bound)))
        self.var_len = var_len = max_nums - min_nums
        bits = np.ceil(np.log2(var_len + 1))

        if DNA_SIZE is None:
            DNA_SIZE = int(np.max(bits))
        self.DNA_SIZE = DNA_SIZE

        self.POP_SIZE = len(nums)
        # POP = np.zeros((*nums.shape, DNA_SIZE))
        # for i in range(nums.shape[0]):
        #     for j in range(nums.shape[1]):
        #         num = int(round((nums[i, j] - bound[j][0]) * ((2 ** DNA_SIZE) / var_len[j])))
        #         POP[i, j] = [int(k) for k in ('{0:0' + str(DNA_SIZE) + 'b}').format(num)]
        # self.POP = POP
        self.POP = nums

        self.copy_POP = nums.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func
        # self.importance = imp

    # def translateDNA(self):
    #     W_vector = np.array([2 ** i for i in range(self.DNA_SIZE)]).reshape((self.DNA_SIZE, 1))[::-1]
    #     binary_vector = self.POP.dot(W_vector).reshape(self.POP.shape[0:2])
    #     for i in range(binary_vector.shape[0]):
    #         for j in range(binary_vector.shape[1]):
    #             binary_vector[i, j] /= ((2 ** self.DNA_SIZE) / self.var_len[j])
    #             binary_vector[i, j] += self.bound[j][0]
    #     return binary_vector
    def get_fitness(self, non_negative=False):
        # results = self.func(*np.array(list(zip(*self.translateDNA()))))
        result = [self.func(self.POP[i]) for i in range(len(self.POP))]
        if non_negative:
            min_fit = np.min(result, axis=0)
            result -= min_fit
        return result


    def select(self):
        fitness = self.get_fitness()
        fit = [item[0] for item in fitness]

        # print(fit)
        # print(np.arange(self.POP.shape[0]))
        # print(self.POP.shape[0])
        # print(fit / np.sum(fit))
        # print(np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0], replace=True,
        #                        p=fit / np.sum(fit)))

        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0], replace=True,
                                             p=fit / np.sum(fit))]
        # for s in self.POP:
        #     print(s)
        # quit()



    def crossover(self):
        for people in self.POP:

            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                cross_points = np.random.randint(0, len(self.bound))
                end_points = np.random.randint(0, len(self.bound) - cross_points)
                people[cross_points:end_points] = self.POP[i_, cross_points:end_points]



    def mutate(self):
        for people in self.POP:
            for point in range(self.DNA_SIZE):
                if np.random.rand() < self.mutation:
                    # var[point] = 1 if var[point] == 0 else 1
                    people[point] = np.random.randint(self.bound[point][0], self.bound[point][1])

    def evolution(self):
        self.select()
        self.crossover()
        self.mutate()


