# check palindrome
from collections import deque


def is_palindrome(nbr: int) -> bool:
    lst: list[int] = [int(digit) for digit in str(nbr)]
    if nbr < 0:
        return False
    if len(lst) == 1:
        return True
    for i in range(len(lst)//2):
        first = lst[i]
        last = lst[-i - 1]
        print(first, last)
        if first != last:
            return False
    return True

# longest comon prefix in an array of string
def longest_common_prefix(strs: list[str]) -> str:
    new_strs: list = sorted(strs)
    min_str: str = new_strs[0]
    common_prefix: str = ""
    for i in range(len(min_str)):
        if has_all(min_str[i], i, new_strs[1:]):
            common_prefix += min_str[i]
        else:
            return common_prefix
    return common_prefix

def has_all(chrs: str, i: int, strs: list) -> bool:
    has: bool = True
    for st in strs:
        if st[i] != chrs:
            has = False
    return has

# valid parentheses
def is_valid(s: str) -> bool:
    if len(s) <= 1:
        return False
    dq: deque = deque()
    for char in s:
        print(dq, char)
        if char in ['(', '{', '[' ]:
            dq.append(char)
        elif char == ')':
            if len(dq) == 0 or dq.pop() != '(':
                return False
        elif char == ']':
            if len(dq) == 0 or dq.pop() != '[':
                return False
        elif char == '}':
            if len(dq) == 0 or dq.pop() != '{':
                return False
    return len(dq) == 0




def main():
    print(is_valid('()[]{}'))

if __name__ == "__main__":
    main()