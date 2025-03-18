def is_different(a:str,b:str)->bool:
    if "~"+a == b or "~"+b == a:
        return True
    return False

def num_to_letter(num):
    return chr(ord('a') + num)

def list_to_str(lst):
    return ",".join(lst)



def from_str_get_num(s: str) -> int:
    digits = ''.join([c for c in s if c.isdigit()])
    return digits if digits else ["0","0"]

def parse_str(input_str):
    result = []

    paren_count = 0
    current_clause = ""
    
    i = 0
    while i < len(input_str):
        c = input_str[i]
        

        if c == '(':
            paren_count += 1
            current_clause += c
        elif c == ')':
            paren_count -= 1
            current_clause += c

            if paren_count == 0 and current_clause.startswith('('):
                inner_str = current_clause[1:-1]
                inner_parts = []
                
                inner_paren_count = 0
                inner_part = ""
                
                for j in range(len(inner_str)):
                    ch = inner_str[j]
                    if ch == '(':
                        inner_paren_count += 1
                    elif ch == ')':
                        inner_paren_count -= 1
                    elif ch == ',' and inner_paren_count == 0:
                        inner_parts.append(inner_part.strip())
                        inner_part = ""
                        continue
                    inner_part += ch
                
                if inner_part:
                    inner_parts.append(inner_part.strip())
                
                result.append(inner_parts)
                current_clause = ""
        elif c == ',' and paren_count == 0:
            if current_clause:
                result.append([current_clause.strip()])
                current_clause = ""
        else:
            current_clause += c
        
        i += 1
    
    if current_clause:
        result.append([current_clause.strip()])
    
    return result
    


import re

def replace_with_dict(input_string, replacement_dict):
    
    sorted_keys = sorted(replacement_dict.keys(), key=len, reverse=True)
    
    pattern = '|'.join(map(re.escape, sorted_keys))
    

    def replacer(match):
        return replacement_dict[match.group(0)]
    

    return re.sub(pattern, replacer, input_string)

def from_dict_to_str(d:dict):
    result_str=""
    for key,value in d.items():
        result_str+=f"({key}={value})"
    return result_str