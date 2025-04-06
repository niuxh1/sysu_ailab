import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

class ga_node:
    def __init__(self, path, fitness):
        self.path = path
        self.fitness = fitness
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __gt__(self, other):
        return self.fitness > other.fitness
    
    def __ge__(self, other):
        return self.fitness >= other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness

def read_data(file_path):
    header = 0
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith('NODE_COORD_SECTION'):
                header = i + 1
    
    df = pd.read_csv(file_path,skiprows=header,header=None,sep=' ')
    df = df.dropna()
    data_np= df.to_numpy()
    for i in range(len(data_np)):
        data_np[i][1] = float(data_np[i][1])
        data_np[i][2] = float(data_np[i][2])
    data_np = data_np[:, 1:]
    return data_np

def fitness(data,path):
    plot_new=np.array([data[path[i]] for i in range(len(path))])
    distance = 0.0
    for i in range(len(path)-1):
        distance += np.linalg.norm(plot_new[i] - plot_new[i+1])
    distance += np.linalg.norm(plot_new[-1] - plot_new[0])
    distance = 1/(distance+1e-6)
    return distance

def tournament_selection(selected):
    selected = np.sort(selected)[::-1]
    return selected[0]

def select_population(ga_list,tournament_size):
    selected = np.random.choice(ga_list, size=tournament_size, replace=False)
    selected = np.array([ga_node(select.path, select.fitness) for i,select in enumerate(selected)], dtype=object)
    return tournament_selection(selected)


def crossover(parent1, parent2):

    i= np.random.randint(0, len(parent1))
    j= np.random.randint(0, len(parent1))
    while i == j:
        j= np.random.randint(0, len(parent1))
    if i > j:
        i, j = j, i
    """
    i 是起点，j是终点
    """
    dict1= {}
    for k in range(i,j+1):
        dict1[parent2[k]] = parent1[k]
    child=np.zeros((len(parent1),), dtype=int)
    for k in range(len(parent1)):
        if k < i or k > j:
            child[k] = parent1[k]
        else:
            child[k]=dict1[parent2[k]]
    return child


def crossover_population(data,ga_list,population=100,crossover_rate = 0.8):
    new_ga_list =  np.zeros((population,), dtype=object)
    for i in range(population):
        if np.random.rand() < crossover_rate:
            parent1 = ga_list[np.random.randint(population)]
            parent2 = ga_list[np.random.randint(population)]
            child_path = crossover(parent1.path, parent2.path)
            new_ga_list[i] = ga_node(child_path, fitness(data, child_path))
        else:
            new_ga_list[i] = ga_list[np.random.randint(population)]
    return new_ga_list

def mutation(path, mutation_rate=0.1):
    for i in range(len(path)):
        if np.random.rand() < mutation_rate:
            i = np.random.randint(0, len(path))
            j = np.random.randint(0, len(path))
            while i == j:
                j = np.random.randint(0, len(path))
            if i > j:
                i, j = j, i
            """
            倒置变异
            """
            path[i:j+1] = path[i:j+1][::-1]
    return path

def mutation_population(data,ga_list,population=100, mutation_rate=0.1):
    new_go_list= np.zeros((population,), dtype=object)
    for i in range(population):
        new_path = mutation(ga_list[i].path, mutation_rate)
        new_go_list[i] = ga_node(new_path, fitness(data, new_path))
    return new_go_list

def adaptive_mutation_rate(iteration, max_iterations = 10000):
    """非线性退火函数 - 前期高探索，后期慢收敛"""
    initial_rate = 0.3  
    final_rate = 0.001  
    
    alpha = 4.0  
    progress = iteration / max_iterations
    if iteration % 500 == 0 and iteration > 0:
        return initial_rate * 0.5  
        
    return final_rate + (initial_rate - final_rate) * np.exp(-alpha * progress) 

def partial_reset(ga_list, data, reset_percentage=0.3):

    population = len(ga_list)
    reset_count = int(population * reset_percentage)
    

    indices = np.argsort([node.fitness for node in ga_list])
    keep_indices = indices[-(population-reset_count):]
    

    new_paths = np.array([np.random.permutation(len(data)) for _ in range(reset_count)])
    new_nodes = np.array([ga_node(path, fitness(data, path)) for path in new_paths], dtype=object)
    
    new_ga_list = np.zeros(population, dtype=object)
    new_ga_list[:reset_count] = new_nodes
    new_ga_list[reset_count:] = ga_list[keep_indices]
    
    return new_ga_list

def calculate_diversity(ga_list):

    fitnesses = np.array([node.fitness for node in ga_list])
    return np.std(fitnesses) / np.mean(fitnesses)




def ga_algorithm(file_path,population=100,tournament_size=5,generations=10000,elite_size=10):
    data=read_data(file_path)
    paths =np.array([np.random.permutation(len(data)) for _ in range(population)])
    ga_list = np.array([ga_node(path, fitness(data, path)) for path in paths], dtype=object)
    min_distances=np.zeros((generations,), dtype=float)
    for i in range(generations):

        elite_indices = np.argsort([node.fitness for node in ga_list])[-elite_size:]
        elites = ga_list[elite_indices].copy()


        min_distances[i]=1/np.max(ga_list).fitness
        """
        gogogo出发喽
        开始选择
        """
        new_ga_list = np.zeros((population,), dtype=object)
        for j in range(population):
            select = select_population(ga_list,tournament_size)
            new_ga_list[j] =ga_node(select.path, select.fitness)
        ga_list = new_ga_list
        """
        开始踩踩
        """
        ga_list = crossover_population(data,ga_list,population=population)
        """
        太坏了，我孩子不是我的
        """
        current_mutation_rate = adaptive_mutation_rate(i)

        ga_list = mutation_population(data,ga_list,population=population, mutation_rate=current_mutation_rate)
        for k, elite in enumerate(elites):
            ga_list[k] = elite
        if i % 50 == 0:
            ga_list = partial_reset(ga_list, data, reset_percentage=0.3)
        if i % 1000 == 0:
            print(f"Generation {i}: Best fitness = {min_distances[i]}")
        if i == generations - 1:
            print(f"Generation {i+1}: Best fitness = {min_distances[i]}")
        
    return np.min(min_distances)

        


ga_algorithm(r"tsp\qa194.tsp",population=100,tournament_size=5,generations=100000,elite_size=10)