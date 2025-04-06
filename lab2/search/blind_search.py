from typing import Optional
import sys
from copy import deepcopy

sys.path.append("c:/Users/ASUS/Desktop/深度学习/lab1")
from utils import find_value

class dfs:
    def __init__(self,graph:list[list[int]]):
        self.graph=graph
        try:
            self.start=find_value(graph,2)
        except ValueError as e:
            print(e)
        try:
            self.end=find_value(graph,3)
        except ValueError as e:
            print(e)
        self.edge={"x_start":0,
                   "x_end":len(self.graph[0])-1,
                   "y_start":0,
                   "y_end":len(self.graph)-1
                   }
                
    def is_in_graph(self,node:list[int])->bool:
        return 0 <= node[0] < len(self.graph) and 0 <= node[1] < len(self.graph[0])

    def _dfs(self, node:list[int], visited=None):
        if visited is None:
            visited = []
        
        if not self.is_in_graph(node):
            return []
        
        if self.graph[node[0]][node[1]] == 1 or any(n[0] == node[0] and n[1] == node[1] for n in visited):
            return []
        
        current_visited = deepcopy(visited)
        current_visited.append(node)
        
        if self.graph[node[0]][node[1]] == 3:
            return current_visited
        
        directions = [[0,1], [1,0], [0,-1], [-1,0]]
        
        for direction in directions:
            new_node = [node[0] + direction[0], node[1] + direction[1]]
            result = self._dfs(new_node, current_visited)
            if result:  
                return result
        
        return []
    
    def __call__(self)->list[list[int]]:
        return self._dfs(self.start)
    

class bfs:
    def __init__(self,graph:list[list[int]]):
        self.graph=graph
        try:
            self.start=find_value(graph,2)
        except ValueError as e:
            print(e)
        try:
            self.end=find_value(graph,3)
        except ValueError as e:
            print(e)
        self.edge={"x_start":0,
                   "x_end":len(self.graph[0])-1,
                   "y_start":0,
                   "y_end":len(self.graph)-1
                   }
                
    def is_in_graph(self,node:list[int])->bool:
        return 0 <= node[0] < len(self.graph) and 0 <= node[1] < len(self.graph[0])

    def _bfs(self):
        parent = {}
        queue = [self.start.copy()]
        visited = [self.start.copy()]
        
        while queue:
            node = queue.pop(0)
            
            if self.graph[node[0]][node[1]] == 3:
                path = []
                current = node
                while True:
                    path.append(current)
                    if current[0] == self.start[0] and current[1] == self.start[1]:
                        break
                    current = parent[(current[0], current[1])]
                path.reverse()
                return path
            
            directions = [[0,1], [1,0], [0,-1], [-1,0]]
            
            for direction in directions:
                new_node = [node[0] + direction[0], node[1] + direction[1]]
                
                if not self.is_in_graph(new_node):
                    continue
                    
                if self.graph[new_node[0]][new_node[1]] == 1:
                    continue
                    
                if any(n[0] == new_node[0] and n[1] == new_node[1] for n in visited):
                    continue
                
                parent[(new_node[0], new_node[1])] = node.copy()
                visited.append(new_node.copy())
                queue.append(new_node.copy())
                    
        return []
    
    def __call__(self)->list[list[int]]:
        return self._bfs()

if __name__=="__main__":
    graph =[]
    while True:
        line=input()
        if not line:
            break
        row=[]
        for char in line:
            if char == "S":
                row.append(2)
            elif char == "E":
                row.append(3)
            else:
                row.append(int(char))
        graph.append(row)
    dfs_=dfs(graph)
    print(dfs_())
    bfs_=bfs(graph)
    print(bfs_())
