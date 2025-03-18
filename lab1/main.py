from utils import *
from copy import deepcopy

class FOL:
    
    def __init__(self,is_not,predicate,args):
        self.is_not = is_not
        self.predicate = predicate
        self.args = args
    
    def __str__(self):
        return ("" if self.is_not else "~") + self.predicate + '(' + list_to_str(self.args) + ')'
    
class solution_atom:
    def __init__(self,is_premise:bool,mgu_dict:dict,front:int,front_letter:int,end:int,end_letter:int,step_fol=list[FOL]):
        self.mgu_dict = mgu_dict
        self.front = front+1
        self.end = end+1
        self.is_premise = is_premise
        self.front_letter = front_letter
        self.end_letter = end_letter
        self.step_fol = step_fol
    def __str__(self):
        if self.is_premise:
            return f"({FOL_list_to_str(self.step_fol)[:-1]})"
        else:
            dict_str="("
            for key,value in self.mgu_dict.items():
                dict_str+=key+"="+value+" "
            dict_str=dict_str[:-1]
            dict_str+= ")" if len(self.mgu_dict) > 0 else ""
            FOL_str=FOL_list_to_str(self.step_fol)[:-1]
            return f"R[{self.front}{num_to_letter(self.front_letter) if self.front_letter >= 0 else ""},{self.end}{num_to_letter(self.end_letter) if self.end_letter >= 0 else ""}]{dict_str} = ({FOL_str})"
        
def list_to_FOL(list):
    FOL_list=[]
    for i in range(len(list)):
        FOL_list_i=[]
        for j in range(len(list[i])):
            is_not = (list[i][j][0] != '~')
            start_bracket = list[i][j].index('(')
            end_bracket = list[i][j].index(')')
            arg = list[i][j][start_bracket+1:end_bracket].split(',')
            predicate =list[i][j][:start_bracket] if is_not else list[i][j][1:start_bracket]
            FOL_list_i.append(FOL(is_not,predicate,arg))
        FOL_list.append(FOL_list_i)
    return FOL_list

def FOL_list_to_str(FOL_list):
    result_str=""
    for i in range(len(FOL_list)):
        result_str+=str(FOL_list[i])+","
    return result_str

def not_in(atom:list[FOL],kb:list[list[FOL]]):
    for i in range(len(kb)):
        if atom == kb[i]:
            return False
    return True

def resolution_algorithm(kb: list[list[FOL]]):
    def run_algorithm(atom1:list[FOL],atom2:list[FOL],atom1_index:int,atom2_index:int,kb:list[list[FOL]],resolution:list[solution_atom]):
        for i in range(len(atom1)):
            for j in range(len(atom2)):
                if atom1[i].is_not != atom2[j].is_not and atom1[i].predicate == atom2[j].predicate:
                    if atom1[i].args == atom2[j].args:
                        if not_in(atom1[:i]+atom1[i+1:]+atom2[:j]+atom2[j+1:],kb):  
                            mgu_dict={}
                            kb.append(atom1[:i]+atom1[i+1:]+atom2[:j]+atom2[j+1:])
                            if len(atom1) == 1 and len(atom2) == 1:
                                resolution.append(solution_atom(False,mgu_dict,atom1_index,-1,atom2_index,-1,kb[-1]))
                            elif len(atom1) == 1 and len(atom2) > 1:
                                resolution.append(solution_atom(False,mgu_dict,atom1_index,-1,atom2_index,j,kb[-1]))
                            elif len(atom1) > 1 and len(atom2) == 1:
                                resolution.append(solution_atom(False,mgu_dict,atom1_index,i,atom2_index,-1,kb[-1]))
                            elif len(atom1) > 1 and len(atom2) > 1:
                                resolution.append(solution_atom(False,mgu_dict,atom1_index,i,atom2_index,j,kb[-1]))
                            if len(kb[-1]) == 0:
                                    return True
                    elif atom1[i].args != atom2[j].args:
                        mgu_dict={}
                        can_unify = True
                        list_fix1=deepcopy(atom1)
                        list_fix2=deepcopy(atom2)
                        
                        for k in range(len(atom1[i].args)):
                            if len(atom1[i].args[k]) > len(atom2[j].args[k]):
                                mgu_dict[atom2[j].args[k]]=atom1[i].args[k]
                            elif len(atom1[i].args[k]) < len(atom2[j].args[k]):
                                mgu_dict[atom1[i].args[k]]=atom2[j].args[k]
                            elif atom1[i].args[k] != atom2[j].args[k]:
                                can_unify = False
                                break
                        
                        if can_unify:
                            for var, val in mgu_dict.items():
                                for l in range(len(list_fix1)):
                                    for m in range(len(list_fix1[l].args)):
                                        if list_fix1[l].args[m] == var:
                                            list_fix1[l].args[m] = val
                                for l in range(len(list_fix2)):
                                    for m in range(len(list_fix2[l].args)):
                                        if list_fix2[l].args[m] == var:
                                            list_fix2[l].args[m] = val
                            
                            result = list_fix1[:i] + list_fix1[i+1:] + list_fix2[:j] + list_fix2[j+1:]
                            if not_in(result, kb):
                                kb.append(result)
                                if len(atom1) == 1 and len(atom2) == 1:
                                    resolution.append(solution_atom(False,mgu_dict,atom1_index,-1,atom2_index,-1,kb[-1]))
                                elif len(atom1) == 1 and len(atom2) > 1:
                                    resolution.append(solution_atom(False,mgu_dict,atom1_index,-1,atom2_index,j,kb[-1]))
                                elif len(atom1) > 1 and len(atom2) == 1:
                                    resolution.append(solution_atom(False,mgu_dict,atom1_index,i,atom2_index,-1,kb[-1]))
                                elif len(atom1) > 1 and len(atom2) > 1:
                                    resolution.append(solution_atom(False,mgu_dict,atom1_index,i,atom2_index,j,kb[-1]))
                                if len(kb[-1]) == 0:
                                    return True
        return False


                                            
    resolution=[]
    for i in range(len(kb)):
        resolution.append(solution_atom(True,{},-1,-1,-1,-1,kb[i]))
    for i in range(len(kb)):
        for j in range(len(kb)):
            for k in range(i+1,len(kb)):
                if run_algorithm(kb[j],kb[k],j,k,kb,resolution):
                    return resolution
    return None

def dfs_del(resolution: list[solution_atom], index: int) -> list[solution_atom]:

    used = [False] * len(resolution)

    index_map = {}

    def dfs_mark(idx):
        if used[idx] or resolution[idx].is_premise:
            return
        used[idx] = True
 
        if not resolution[idx].is_premise:
            dfs_mark(resolution[idx].front - 1)
            dfs_mark(resolution[idx].end - 1)

    dfs_mark(index)
    

    for i in range(len(resolution)):
        if resolution[i].is_premise:
            used[i] = True

    new_resolution = []
    new_index = 1
    for i in range(len(resolution)):
        if used[i]:
            index_map[i+1] = new_index
            new_resolution.append(resolution[i])
            new_index += 1

    for i in range(len(new_resolution)):
        if not new_resolution[i].is_premise:
            new_resolution[i].front = index_map[new_resolution[i].front]
            new_resolution[i].end = index_map[new_resolution[i].end]
    
    return new_resolution

if __name__ == '__main__':
    # kb = [['GradStudent(sue)',], ['~GradStudent(x)', 'Student(x)'], ['~Student(x)', 'HardWorker(x)'],
    #    ['~HardWorker(sue)',]]
    # kb=list_to_FOL(kb)

    kb = [
    ['A(tony)'], 
    ['A(mike)'], 
    ['A(john)'], 
    ['L(tony,rain)'], 
    ['L(tony,snow)'], 
    ['~A(x)', 'S(x)', 'C(x)'],
    ['~C(y)', '~L(y,rain)'], 
    ['L(z,snow)', '~S(z)'], 
    ['~L(tony,u)', '~L(mike,u)'], 
    ['L(tony,v)', 'L(mike,v)'],
    ['~A(w)', '~C(w)', 'S(w)']
    ]
    kb = list_to_FOL(kb)

    # kb =[['On(aa,bb)'],['On(bb,cc)'],['Green(aa)'],['~Green(cc)'],['~On(x,y)','~Green(x)','Green(y)']]
    # kb = list_to_FOL(kb)
    
    # for i in range(len(kb)):
    #     for j in range(len(kb[i])):
    #         print(kb[i][j],end=',')
    #     print()
    # solution_atom1 = solution_atom(False,{'x': 'sue'}, 0,-1, 3,3, kb[1])
    # print(solution_atom1)
    resolution = resolution_algorithm(kb)
    if resolution == None:
        print("No solution")
    else:
        # 递归删除未使用的推理步骤，并更新对应标号
        resolution = dfs_del(resolution, len(resolution) - 1)
        for i in range(len(resolution)):
            print(f"{i+1}:{resolution[i]}")