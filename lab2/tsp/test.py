import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class GeneticAlgTSP:
    def __init__(self, filename: str, population_size: int = 100):
        """
        初始化TSP遗传算法求解器
        
        Args:
            filename: TSP数据文件名
            population_size: 种群大小
        """
        self.filename = filename
        self.population_size = population_size
        self.cities = self._read_tsp_file(filename)
        self.num_cities = len(self.cities)
        self.population = self._init_population()
        self.best_solution = None
        self.best_distance = float('inf')
        
    def _read_tsp_file(self, filename: str) -> np.ndarray:
        """读取TSP文件中的城市坐标"""
        file_path = filename if '/' in filename or '\\' in filename else f"tsp/{filename}"
        header = 0
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if line.startswith('NODE_COORD_SECTION'):
                    header = i + 1
                    break
        
        df = pd.read_csv(file_path, skiprows=header, header=None, sep=' ')
        df = df.dropna()
        data_np = df.to_numpy()
        
        # 处理坐标数据
        cities = np.zeros((len(data_np), 2))
        for i in range(len(data_np)):
            cities[i][0] = float(data_np[i][1])
            cities[i][1] = float(data_np[i][2])
            
        return cities
    
    def _init_population(self) -> List[np.ndarray]:
        """初始化种群"""
        return [np.random.permutation(self.num_cities) for _ in range(self.population_size)]
    
    def _calculate_distance(self, route: np.ndarray) -> float:
        """计算路径总距离"""
        total_distance = 0
        for i in range(len(route) - 1):
            city1 = self.cities[route[i]]
            city2 = self.cities[route[i + 1]]
            total_distance += np.linalg.norm(city1 - city2)
        
        # 添加回到起点的距离
        total_distance += np.linalg.norm(self.cities[route[-1]] - self.cities[route[0]])
        return total_distance
    
    def _evaluate_population(self) -> np.ndarray:
        """评估种群中每个个体的适应度"""
        fitness = np.zeros(self.population_size)
        for i, route in enumerate(self.population):
            distance = self._calculate_distance(route)
            fitness[i] = 1.0 / distance  # 适应度为距离的倒数
            
            # 更新最佳解
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_solution = route.copy()
                
        return fitness
    
    def _selection(self, fitness: np.ndarray) -> List[np.ndarray]:
        """使用轮盘赌选择策略选择父代"""
        # 计算选择概率
        total_fitness = np.sum(fitness)
        if total_fitness == 0:
            selection_probs = np.ones(self.population_size) / self.population_size
        else:
            selection_probs = fitness / total_fitness
            
        # 选择父代
        selected_indices = np.random.choice(
            self.population_size, 
            size=self.population_size, 
            p=selection_probs
        )
        return [self.population[i].copy() for i in selected_indices]
    
    def _tournament_selection(self, fitness: np.ndarray, tournament_size: int = 5) -> List[np.ndarray]:
        """锦标赛选择策略"""
        selected = []
        for _ in range(self.population_size):
            # 随机选择tournament_size个个体
            competitors = np.random.choice(self.population_size, size=tournament_size, replace=False)
            # 选择适应度最高的个体
            winner_idx = competitors[np.argmax(fitness[competitors])]
            selected.append(self.population[winner_idx].copy())
        return selected
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用顺序交叉(OX)实现交叉操作"""
        size = len(parent1)
        
        # 随机选择交叉点
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        # 创建子代
        child1 = np.zeros(size, dtype=int) - 1  # 用-1填充
        child2 = np.zeros(size, dtype=int) - 1
        
        # 从父代复制交叉段
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # 填充剩余位置
        self._fill_crossover_child(child1, parent2, start, end)
        self._fill_crossover_child(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_crossover_child(self, child: np.ndarray, parent: np.ndarray, start: int, end: int):
        """填充交叉后的子代"""
        size = len(child)
        # 创建一个包含子代中已有城市的集合
        used = set(child[start:end])
        
        # 从父代中按顺序取出未使用的城市
        idx = end % size
        for city in parent:
            if city not in used:
                child[idx] = city
                used.add(city)
                idx = (idx + 1) % size
                if idx == start:
                    break
    
    def _mutation(self, route: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """变异操作：随机交换城市顺序"""
        if np.random.random() < mutation_rate:
            # 随机选择两个位置并交换
            i, j = np.random.choice(len(route), 2, replace=False)
            route[i], route[j] = route[j], route[i]
        return route
    
    def iterate(self, num_iterations: int = 1000, crossover_rate: float = 0.8, 
                mutation_rate: float = 0.1, tournament_size: int = 5) -> List[int]:
        """
        进行遗传算法迭代
        
        Args:
            num_iterations: 迭代次数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            tournament_size: 锦标赛大小
            
        Returns:
            最优路径
        """
        for gen in range(num_iterations):
            # 评估种群
            fitness = self._evaluate_population()
            
            # 选择
            selected = self._tournament_selection(fitness, tournament_size)
            
            # 创建新种群
            new_population = []
            
            # 精英保留
            elite_idx = np.argmax(fitness)
            new_population.append(self.population[elite_idx].copy())
            
            # 交叉和变异
            for i in range(0, self.population_size - 1, 2):
                if i + 1 < self.population_size:
                    if np.random.random() < crossover_rate:
                        child1, child2 = self._crossover(selected[i], selected[i + 1])
                        new_population.append(self._mutation(child1, mutation_rate))
                        if len(new_population) < self.population_size:
                            new_population.append(self._mutation(child2, mutation_rate))
                    else:
                        new_population.append(selected[i].copy())
                        if len(new_population) < self.population_size:
                            new_population.append(selected[i + 1].copy())
                else:
                    new_population.append(selected[i].copy())
            
            # 更新种群
            self.population = new_population
            
            # 打印进度
            if (gen + 1) % 1000 == 0:
                print(f"Generation {gen + 1}/{num_iterations}, Best distance: {self.best_distance:.2f}")
        
        # 返回最优路径（从0开始的城市索引）
        return [int(city) for city in self.best_solution]
    
    def plot_route(self, route=None):
        """可视化TSP路径"""
        if route is None:
            route = self.best_solution
        
        plt.figure(figsize=(10, 6))
        
        # 绘制城市点
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', s=40)
        
        # 绘制路径
        for i in range(len(route)):
            j = (i + 1) % len(route)
            plt.plot([self.cities[route[i]][0], self.cities[route[j]][0]],
                     [self.cities[route[i]][1], self.cities[route[j]][1]], 'r-')
        
        plt.title(f'TSP Route (Total distance: {self._calculate_distance(route):.2f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
        
    def get_best_distance(self) -> float:
        """获取最佳距离"""
        return self.best_distance

# 使用示例
if __name__ == "__main__":
    # 解决Djibouti的38个城市TSP问题
    tsp_solver = GeneticAlgTSP(r"tsp\qa194.tsp", population_size=100)
    best_route = tsp_solver.iterate(num_iterations=100000, crossover_rate=0.8, 
                                    mutation_rate=0.1, tournament_size=5)
    
    print(f"Best distance found: {tsp_solver.get_best_distance():.2f}")
    
    # 可视化最优路径
    tsp_solver.plot_route()