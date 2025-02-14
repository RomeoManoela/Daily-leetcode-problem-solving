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

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# Merge two sorted list
def merge_two_sorted_list(list1: ListNode, list2: ListNode) -> ListNode  :
    dummy = ListNode()
    current = dummy
    while list1 and list2:
        if list1.val < list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    if list1:
        current.next = list1
    elif list2:
        current.next = list2

    return dummy.next

# remove duplicate from sorted list
def remove_duplicates(self, nums: list[int]) -> int:
    if not nums:
        return 0
    i: int = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1
# remove element in an array
def remove_element( nums: list[int], val: int) -> int:
    result: list[int] = []
    i: int = 0
    for num in nums:
        if num != val:
            result.append(num)
            i = i + 1
    nums[:] = result
    return i




def main():
    ns = [0, 1, 4, 3, 2, 2, 1]
    print(ns)
    print(remove_element(ns, 1))
    print(ns)

if __name__ == "__main__":
    main()