import numpy as np


# classifier_name = 'Random_Forest_standard_unfair.pkl'
# model = joblib.load(classifier_name)

class GA:
    # input:
    #     nums: m * n  n is nums_of x, y, z, ...,and m is population's quantity
    #     bound:n * 2  [(min, nax), (min, max), (min, max),...]
    #     DNA_SIZE is binary bit size, None is auto
    def __init__(self, nums, bound, func, datasetName, sensitive_param, DNA_SIZE=None, cross_rate=0.8, mutation=0.003,
                 mutation_positive=None,
                 mutation_negative=None, ):
        nums = np.array(nums)
        bound = np.array(bound)
        # print(bound)

        # self.FILE_NAME = "./Numpy/Log_File_{}_{}_rev.txt".format(datasetName, sensitive_param)
        # self.SAMPLE_FILE_NAME = "{}_{}_rev.npy".format(datasetName, sensitive_param)

        self.bound = bound
        self.sensitive_param = sensitive_param
        self.EPOCHS = 0
        min_nums, max_nums = np.array(list(zip(*bound)))
        # print(bound)
        # print(*bound)
        # print(zip(*bound))
        # print(list(zip(*bound)))
        # print(min_nums, max_nums)
        self.var_len = var_len = max_nums - min_nums

        bits = np.ceil(np.log2(var_len + 1))
        if DNA_SIZE is None:
            DNA_SIZE = int(np.max(bits))

        self.DNA_SIZE = DNA_SIZE

        self.POP_SIZE = len(nums)

        self.total_samples = set()
        self.total_samples_list = []
        self.disc_samples = set()
        self.disc_samples_list = []

        self.POP = nums
        self.SIZE = self.POP.shape[0]
        self.copy_POP = nums.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func
        self.mutation_positive = mutation_positive
        self.mutation_negative = mutation_negative
        self.IsDisc = np.array([])
        # self.importance = imp

    def get_fitness(self, non_negative=False):

        fitness_list = []
        disc_list = []
        for i in range(len(self.POP)):
            f, p = self.func(self.POP[i])

            fitness_list.append(f)
            disc_list.append(p)

            sample = [int(x) for x in self.POP[i]]

            sample = np.reshape(sample, (1, -1))
            self.total_samples.add(tuple(map(tuple, sample)))
            if p == 1:
                # print(sample)
                self.disc_samples.add(tuple(map(tuple, sample)))
                # print(sample)
                # quit()

        self.IsDisc = np.array(disc_list)

        if non_negative:
            min_fit = np.min(fitness_list, axis=0)
            fitness_list -= min_fit
        return fitness_list

    def select(self):
        fitness = self.get_fitness()
        fit = [item for item in fitness]

        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.SIZE, replace=True,
                                             p=fit / np.sum(fit))]
        self.get_fitness()

    def mutate(self):
        for index in range(len(self.POP)):
            if self.IsDisc[index] == 0:

                mutation_index = np.random.choice(np.arange(self.DNA_SIZE),
                                                  size=np.random.randint(1, 5, 1),
                                                  replace=False, p=self.mutation_negative)

                for a in mutation_index:
                    self.POP[index][a] = np.random.randint(self.bound[a][0], self.bound[a][1])
            else:

                mutation_index = np.random.choice(np.arange(self.DNA_SIZE),
                                                  size=np.random.randint(1, 5, 1),
                                                  replace=False, p=self.mutation_positive)

                for a in mutation_index:
                    self.POP[index][a] = np.random.randint(self.bound[a][0], self.bound[a][1])

    def crossover(self):
        newList = self.POP.tolist()
        for people in self.POP:
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                cross_points = np.random.randint(0, len(self.bound))
                end_points = np.random.randint(0, len(self.bound) - cross_points)

                new_sample0 = np.array(people).tolist()
                new_sample1 = np.array(self.POP[i_]).reshape(-1).tolist()

                new_sample0[cross_points:cross_points + end_points], new_sample1[
                                                                     cross_points:cross_points + end_points] = new_sample1[
                                                                                                               cross_points:cross_points + end_points], new_sample0[
                                                                                                                                                        cross_points:cross_points + end_points]

                newList.append(new_sample0)
                newList.append(new_sample1)

        self.POP = np.array(newList)

    def evolution(self):


        self.select()
        self.mutate()
        self.crossover()

        '''
            Record 
        '''
        # self.EPOCHS += 1

        # if self.EPOCHS % 100 == 0 or self.EPOCHS % 100 == 1:
        #     SAMPLE_FILE_NAME = "./Numpy/SAMPLE_ALL_{}_".format(self.EPOCHS) + self.SAMPLE_FILE_NAME
        #     SAMPLE_DISC_NAME = "./Numpy/SAMPLE_DIS_{}_".format(self.EPOCHS) + self.SAMPLE_FILE_NAME
        #     new_list = []
        #     print(self.IsDisc.sum())
        #     for i in range(len(self.IsDisc)):
        #         if self.IsDisc[i] == 1:
        #             new_list.append(self.POP[i])
        #             # print(new_list)
        #     # print(SAMPLE_FILE_NAME)
        #     # print(new_list)
        #     # quit()
        #
        #     np.save(SAMPLE_FILE_NAME, np.array(self.POP))
        #     np.save(SAMPLE_DISC_NAME, np.array(new_list))
        #     f = open(self.FILE_NAME, "a")
        #     f.write(SAMPLE_FILE_NAME + " ({}) be saved \n".format(np.array(self.POP).shape))
        #     f.write(SAMPLE_DISC_NAME + " ({}) be saved \n".format(np.array(new_list).shape))
        #     f.close()
