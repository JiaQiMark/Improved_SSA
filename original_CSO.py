import numpy as np
import math
import random
import matplotlib.pyplot as plt


class Chicken:
    # Defining Constructor
    def __init__(self, fitness_function, range_l, range_u, Dn):
        # For The Initial Setup , Later Will Store the Next Position
        # which will be evaluated for Fitness and Storing in the string or Not
        self.range_l = range_l
        self.range_u = range_u
        self.next_position = np.random.uniform(self.range_l, self.range_u, Dn)
        self.original_position = self.next_position
        self.fitness = -1  # Inititally Not Evaluating the Fitness
        self.group = -1  # Inirially Not Evaluating Any Group
        self.species_name = "none"  # Later Will change to Rooster, Chicken or Hen
        self.fitness_function = fitness_function
        self.Dn = Dn
    '''
        A Function Which is Subjected to Get The Fitness Count of the Hen based on The Criteria
        The Function is Called Twice , First When The Assignment for Both Rooster and Chickens is done and At Last when 
        All The Fitness is compared and the best is among to be chosen
    '''

    def evaluate(self):
        self.original_position = self.next_position
        self.fitness = self.fitness_function(self.original_position)

    '''
        Group of Functions which need to update the position of Chickens.
        Note : The Position will be first stored in a different property ie next_position will store the next address . Moving on the lane , The Fitness Count Obtained from new Generation will help in Updating the solution
    '''

    '''
        All the Functions will take in a parameter as The Number of Groups the Population is Divied into , For Example , If the Total Population is 10 , The best suited Group will be 10/5 , ie 2. All the Roosters will be then updated to the count of the following Appropriate Distribution
    '''

    def update_location_rooster(self, number_of_groups, rooster):  # Integer , Class Rooster
        # Example as Like if the Population is divided into 2 Groups ,
        # and then Total Option is Limited to either 0 or 1 , 0 for the First Group and 1 For Other Group
        random_integer_of_groups = np.random.randint(0, number_of_groups)

        while rooster[random_integer_of_groups].group == self.group:
            ## Checking If It doesn't Belong to the same Group
            random_integer_of_groups = np.random.randint(0, number_of_groups)

        ### Evaluating The Equation According to The Algorithm

        ## Initalizing Sigma
        sigma_square = 0
        e = 0.000000000000000000000000000000000001
        if rooster[random_integer_of_groups].group != self.group:
            if rooster[random_integer_of_groups].fitness >= self.fitness:
                sigma_square = 1
            else:
                sigma_square = np.exp((rooster[random_integer_of_groups].fitness - self.fitness)
                                      / (np.abs(self.fitness) + e))

        # Create Gaussian Distribution  with Mean 0 and Standard Deviation is sigma_sqare
        random_distribution = np.random.normal(0, sigma_square)

        '''
            We are Only Updating The Next Position , And Not the Original Position , 
            Because the Update is Valid only when The Original Fitness is found lowered to The Mutated Fitness
        '''

        for index in range(0, self.Dn):
            self.next_position[index] = self.original_position[index] * (1 + random_distribution)
            self.next_position[index] = np.clip(self.next_position[index], self.range_l, self.range_u)

    def update_location_hen(self, number_of_groups, rooster):  # Integer , Class Rooster
        fitness_rooster_1 = None
        fitness_rooster_2 = None
        position_rooster_1 = None
        position_rooster_2 = None
        for index in range(0, number_of_groups):
            if rooster[index].group == self.group:
                position_rooster_1 = rooster[index].original_position  # Same Group Rooster Position
                fitness_rooster_1 = rooster[index].fitness  # Same Group Rooster Health

        random_integer_of_groups = np.random.randint(0, number_of_groups)
        while rooster[random_integer_of_groups].group == self.group:
            ## More not Getting the same Rooster Group
            random_integer_of_groups = np.random.randint(0, number_of_groups)

        if rooster[random_integer_of_groups].group != self.group:
            # Some K Rooster Index
            position_rooster_2 = rooster[random_integer_of_groups].original_position
            # Some K Rooster's Fitness
            fitness_rooster_2 = rooster[random_integer_of_groups].fitness

        fitness_current_hen = self.fitness  # Fitness of Current Hen
        position_current_hen = self.original_position  # Position of Current Hen
        e = 0.000000000000000000000000000000000001  # Defining the Smallest Constant

        # Defining S1 and S2 For The Parameters Listed
        S1 = np.exp((fitness_current_hen - fitness_rooster_1) / (np.abs(fitness_current_hen) + e))
        S2 = np.exp((fitness_rooster_2 - fitness_current_hen))

        # Defining a Uniform Random Number Between 0 and 1
        uniform_random_number_between_0_and_1 = np.random.rand()

        # Note , Changing the next position and not the original position for Comparing different fitness
        for index in range(0, self.Dn):
            self.next_position[index] = (position_current_hen[index] +
                S1 * uniform_random_number_between_0_and_1 * (position_rooster_1[index] - position_current_hen[index])
              + S2 * uniform_random_number_between_0_and_1 * (position_rooster_2[index] - position_current_hen[index]))
            self.next_position[index] = np.clip(self.next_position[index], self.range_l, self.range_u)

    # A Floating Point Value Between 0 and 2 , Array Containing the Position of the Mother Hen
    def update_location_chick(self, FL, position_of_mother_hen):
        # Getting The Current Chick Position
        position_current_chick = self.original_position

        for index in range(0, self.Dn):
            self.next_position[index] = (position_current_chick[index] + FL * (
                        position_of_mother_hen[index] - position_current_chick[index]))
            self.next_position[index] = np.clip(self.next_position[index], self.range_l, self.range_u)


def implementing_cso(population, 
                     maximum_generation, 
                     search_range_l,
                     search_range_u,
                     Dn, 
                     fitness_function, 
                     individuals_group, 
                     self_update_time, 
                     FL=0.5):
    # Initializing the total number of Groups for the Population ,
    # Appropriate Will be Population in Multiple of 10's and Dividing It in Multiple of 5
    number_of_groups = int(population / individuals_group)
    # print("The Number Of Group The Swarm Is Divided : ", number_of_groups)

    population_list = []  # List Storing the Object of Chicken.

    for index in range(population):
        population_list.append(Chicken(fitness_function, search_range_l, search_range_u, Dn))
        population_list[index].evaluate()

    iteration_test_cases = 0
    optimal_solution_fitness_list = []
    group_list_containing_which_group_belongs = None
    rooster_class = None
    while iteration_test_cases < maximum_generation:
        # update After Every Certain Time
        if iteration_test_cases % self_update_time == 0:
            population_list.sort(key=lambda x: x.fitness, reverse=False)

            # Assigning The Members Equally in a Group
            # Assigning Equal Number of Roosters to Each Group
            rooster_class = population_list[:number_of_groups]
            # Assigning the Last Remaining Classes as Chick
            chicks_class = population_list[-(2 * number_of_groups)]
            hens_class = population_list[-(population - number_of_groups): -(2 * number_of_groups)]

            # Group === Knowing Which Chicken Belongs to Which Group .
            #   Can Either be Done Through Going through each class and getting Group Number it Belongs to.
            group_list_containing_which_group_belongs = np.zeros(population)

            for index in range(number_of_groups):
                population_list[index].species_name = "Rooster"
                population_list[index].group = index
                group_list_containing_which_group_belongs[index] = index

            # Assigning Hens in the Group
            bundary_num_hen_chicks = int(((population - number_of_groups) / 2) + number_of_groups)
            for index in range(number_of_groups, bundary_num_hen_chicks):
                population_list[index].species_name = "Hen"
                population_list[index].group = index % number_of_groups
                group_list_containing_which_group_belongs[index] = population_list[index].group

                population_list[bundary_num_hen_chicks + index - number_of_groups].species_name = "Chick"
                population_list[bundary_num_hen_chicks + index - number_of_groups].group = population_list[index].group
                group_list_containing_which_group_belongs[bundary_num_hen_chicks + index - number_of_groups] = index

            '''
                Based on the Total Population , The Population is divided into Group of 5 lets suppose. 
                Now To each Group , We will have 1 Head Rooster , 2 Hens and 2 Chicks . 
                Now , The Algorithm Validates to the  Position being Updated for each Row 
                and The Validation successfully yields the nature Criteria for Identifying Weak as well as Strong . 
                Thereby Performing Swarm Optimization
            '''

            # In Example of 10 , With 2 Groups [ 0., 0.  ]
            # 每个小组中有几个roosters hens chicks
            roosters_in_each_group_counter = np.zeros(number_of_groups)
            hens_in_each_group_counter = np.zeros(number_of_groups)
            chicks_in_each_group_counter = np.zeros(number_of_groups)

            '''
                To Assign Roosters and Hens And Chickens , We will Try to Randomize as much as Possible , 
                failing of which will indicate the Gain of a particular group every Time , 
                thereby Hampering Our Solution.
            '''

            # for i in range(0, population):
            #    print("Fitness is ", population_list[i].fitness)
            # print("The Roosters Count is : ", roosters_in_each_group_counter, "\n",
            #       "The Hen Count is : ", hens_in_each_group_counter, "\n",
            #       "The Chick Count is ", chicks_in_each_group_counter)
            # print("The Group List Looks like ", group_list_containing_which_group_belongs)

        # It Starts Here!!!!
        for index in range(0, population):
            if population_list[index].species_name == "Rooster":
                population_list[index].update_location_rooster(number_of_groups, rooster_class)
            elif population_list[index].species_name == "Hen":
                population_list[index].update_location_hen(number_of_groups, rooster_class)
            elif population_list[index].species_name == "Chick":
                mother_hen_index = int(group_list_containing_which_group_belongs[index])
                position_of_mother_hen = population_list[mother_hen_index].original_position
                population_list[index].update_location_chick(FL, position_of_mother_hen)

            population_list[index].evaluate()
            # reverse=False  升序排序，从低到高
            population_list.sort(key=lambda x: x.fitness, reverse=False)


        if iteration_test_cases == 0:
            fitness_value = population_list[0].fitness
        else:
            fitness_value = min(population_list[0].fitness, optimal_solution_fitness_list[-1])


        optimal_solution_fitness_list.append(fitness_value)
        # print("CSO", iteration_test_cases, "/", maximum_generation)
        iteration_test_cases += 1

    return optimal_solution_fitness_list





# optimal_solution = implementing_cso(population_size, group_size, max_iterations, 1, 0.5)
# iterations = np.linspace(0, max_iterations-1, len(optimal_solution), dtype=int)
#
# plt.xlabel('iterations')
# plt.ylabel('fitness')
# plt.title('cso')
# plt.yscale('log')
# plt.plot(iterations, optimal_solution)
# plt.show()

