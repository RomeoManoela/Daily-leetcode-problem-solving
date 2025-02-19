# check palindrome
from collections import deque


def is_palindrome(nbr: int) -> bool:
    lst: list[int] = [int(digit) for digit in str(nbr)]
    if nbr < 0:
        return False
    if len(lst) == 1:
        return True
    for i in range(len(lst) // 2):
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
        if char in ["(", "{", "["]:
            dq.append(char)
        elif char == ")":
            if len(dq) == 0 or dq.pop() != "(":
                return False
        elif char == "]":
            if len(dq) == 0 or dq.pop() != "[":
                return False
        elif char == "}":
            if len(dq) == 0 or dq.pop() != "{":
                return False
    return len(dq) == 0


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Merge two sorted list
def merge_two_sorted_list(list1: ListNode, list2: ListNode) -> ListNode:
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
def remove_element(nums: list[int], val: int) -> int:
    result: list[int] = []
    i: int = 0
    for num in nums:
        if num != val:
            result.append(num)
            i = i + 1
    nums[:] = result
    return i


# index of the first occurrence of str2 in str1
def str_str(haystack: str, needle: str) -> int:
    if not needle in haystack:
        return -1

    for i in range(len(haystack) - len(needle) + 1):
        print(haystack[i : i + len(needle)], i)
        if haystack[i : i + len(needle)] == needle:
            return i


# search insert position
def search_insert(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return left


# last word length
def length_of_last_word(s: str) -> int:
    strs: list[str] = s.split(" ")
    print(strs)
    last_word: str = [item for item in strs if item != ""].pop()
    return len(last_word)


# PLus one
def plus_one(digits: list[int]) -> list[int]:
    n: int = len(digits) - 1
    for i in range(n, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    return [1] + digits


def main():
    print(length_of_last_word("romeo  manoela     zadd  "))
    print(plus_one([9]))


if __name__ == "__main__":
    main()
