
from numpy import where,ndarray,array,array_equal,zeros,int8,random
import heapq
import time
from numba import njit


@njit
def find_value(graph: ndarray, value: int) -> int:
    indices = where(graph == value)[0]
    return indices[0]



@njit
def h_manhattan_numba(graph, goal_positions):
    h = 0
    for i in range(16):
        val = graph[i]
        if val == 0:
            continue
        correct_pos = goal_positions[val]
        current_row, current_col = i // 4, i % 4
        correct_row, correct_col = correct_pos // 4, correct_pos % 4
        h += abs(current_row - correct_row) + abs(current_col - correct_col)
    
    return h

@njit
def h_manhattan_with_conflicts_numba(current, goal, goal_positions):
    h1 = h_manhattan_numba(current, goal_positions)
    h2 = 0
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            val = current[idx]
            if val == 0:
                continue
            
            goal_pos = goal_positions[val]
            goal_row, goal_col = goal_pos // 4, goal_pos % 4
            
            if i == goal_row:
                for k in range(j+1, 4):
                    idx2 = i * 4 + k
                    val2 = current[idx2]
                    if val2 != 0:
                        if val2 < 16:  # 防止数组越界
                            goal_pos2 = goal_positions[val2]
                            goal_row2, goal_col2 = goal_pos2 // 4, goal_pos2 % 4
                            if goal_row2 == i and goal_col2 < goal_col:
                                h2 += 2
            
            if j == goal_col:
                for k in range(i+1, 4):
                    idx2 = k * 4 + j
                    val2 = current[idx2]
                    if val2 != 0:
                        if val2 < 16:  # 防止数组越界
                            goal_pos2 = goal_positions[val2]
                            goal_row2, goal_col2 = goal_pos2 // 4, goal_pos2 % 4
                            if goal_col2 == j and goal_row2 < goal_row:
                                h2 += 2
    
    return h1 + h2



class node:
    __slots__ = ['graph', 'parent_node', 'parent', 'g', 'h', 'f', '_hash', 'empty_pos']
    
    def __init__(self, graph, parent_node=None, parent=None, empty_pos=None):
        self.graph = graph
        self.parent_node = parent_node
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
        self._hash = self._calculate_hash()
        
        self.empty_pos = find_value(self.graph, 0)
    
    def _calculate_hash(self):
    # 使用内置tuple哈希可能比位运算更快
        return hash(tuple(self.graph))
    
    def __lt__(self, other):
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        if not isinstance(other, node):
            return False
        return self._hash == other._hash and array_equal(self.graph, other.graph)
    
    def get_neighbors(self):
        neighbours = []
        row, col = self.empty_pos // 4, self.empty_pos % 4
        
        directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            
            if not (0 <= new_row < 4 and 0 <= new_col < 4):
                continue
            
            new_pos = new_row * 4 + new_col
            new_graph = self.graph.copy()
            tile_value = new_graph[new_pos]
            new_graph[self.empty_pos] = tile_value
            new_graph[new_pos] = 0
            neighbours.append(node(new_graph, tile_value, self, new_pos))
        
        return neighbours
# @jitclass 

class astar:
    def __init__(self, graph, max_depth=70):
        self.graph = array(graph, dtype=int8)
        self.answer = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], dtype=int8)
        self.max_depth = max_depth
        self.goal_positions = zeros(16, dtype=int8)
        for i in range(16):
            self.goal_positions[self.answer[i]] = i
    
    def _astar(self):

            
        open_set = []
        open_dict = {}
        close_set = set()
        goal_hash = hash(node(self.answer))


        h_func = lambda state: h_manhattan_with_conflicts_numba(state, self.answer, self.goal_positions)
        

        start_node = node(self.graph, None, None)
        start_node.h = h_func(self.graph)
        start_node.f = start_node.h
        
        node_hash = hash(start_node)
        heapq.heappush(open_set, start_node)
        open_dict[node_hash] = start_node
        
        
        while open_set:
            current = heapq.heappop(open_set)
            current_hash =current._hash
            
            if current_hash in close_set:
                continue
                
            if current_hash in open_dict:
                del open_dict[current_hash]

            if current_hash == goal_hash:
                path = []
                node_ptr = current
                while node_ptr and node_ptr.parent_node is not None:
                    path.append(node_ptr.parent_node)
                    node_ptr = node_ptr.parent
                return path[::-1]
            
            close_set.add(current_hash)

            if current.g >= self.max_depth:
                continue

            for neighbour in current.get_neighbors():
                neighbour_hash = hash(neighbour)

                if neighbour_hash in close_set:
                    continue

                new_g = current.g + 1
                
                if new_g > self.max_depth:
                    continue

                if neighbour_hash in open_dict and open_dict[neighbour_hash].g <= new_g:
                    continue

                neighbour.g = new_g
                neighbour.h = h_func(neighbour.graph)
                neighbour.f = neighbour.g + h_func(neighbour.graph)
                
                open_dict[neighbour_hash] = neighbour
                heapq.heappush(open_set, neighbour)
                


    def __call__(self):
        return self._astar()

if __name__ == "__main__":
    start_time = time.time()
    
    graph = array([0,5,15,14,7,9,6,13,1,2,12,10,8,11,4,3], dtype=int8)
    
    
    
    a = astar(graph, 65)
    

    result = a()
    print(f"找到解决方案，总步数: {len(result)}")
    print(result)

    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.2f}秒")