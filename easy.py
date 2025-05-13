import heapq
from collections import deque
from typing import Optional, Any


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def is_palindrome(nbr: int) -> bool:
    if nbr < 0:
        return False
    str_nbr: str = str(nbr)  # convert int to str
    for i in range(len(str_nbr) // 2):
        first = str_nbr[i]
        last = str_nbr[-i - 1]
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


# remove duplicate from a sorted list
def remove_duplicates(self, nums: list[int]) -> int:
    if not nums:
        return 0
    i: int = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1


# remove an element in an array
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
def str_str(haystack: str, needle: str) -> int | None:
    if not needle in haystack:
        return -1

    for i in range(len(haystack) - len(needle) + 1):
        print(haystack[i : i + len(needle)], i)
        if haystack[i : i + len(needle)] == needle:
            return i
        return None
    return None


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


# Add binary
def add_binary(a: str, b: str) -> str:
    a = int("0b" + a, 2)
    b = int("0b" + b, 2)
    res = bin(a + b)
    return res[2:]


# sqrt(x)
def my_sqrt(x: int) -> int:
    if x == 0:
        return 0
    left, right = 1, x
    while left <= right:
        mid = (left + right) // 2
        if mid * mid == x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        else:
            right = mid - 1
    return right


# climbStairs: it took me 1h to solve,
# I didn't realize that the problem is a fibonacci sequence
def climb_stairs(n: int) -> int:
    if n == 1:
        return 1
    prev1, prev2 = 1, 2
    for _ in range(3, n + 1):
        current = prev1 + prev2
        prev1, prev2 = prev2, current
    return prev2


# remove duplicate from a sorted list
def delete_duplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    unique: set[int] = set()
    current: ListNode = head
    while current:
        if current.val not in unique:
            unique.add(current.val)
        else:
            before = current


# merge sorted list
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    last = m + n - 1
    m = m - 1
    n = n - 1

    while n >= 0:
        if m >= 0 and nums1[m] > nums2[n]:
            nums1[last] = nums1[m]
            m -= 1
        else:
            nums1[last] = nums2[n]
            n -= 1
        last -= 1


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# traverse a binary search tree from left to right
def inorder_traversal(root: Optional[TreeNode]) -> list[int]:
    result: list[int] = []

    def dfs(node: Optional[TreeNode]) -> None:
        if not node:
            return

        dfs(node.left)
        result.append(node.val)
        dfs(node.right)

    dfs(root)
    return result


def is_symmetric(root: Optional[TreeNode]) -> bool:
    if not root:
        return True

    def is_mirror(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val != right.val:
            return False
        return is_mirror(left.left, right.right) and is_mirror(left.right, right.left)

    return is_mirror(root.left, root.right)


def max_depth(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    return 1 + max(self.max_depth(root.left), self.max_depth(root.right))


# sorted array to BST
def sorted_array_to_bst(nums: list[int]) -> Optional[TreeNode]:
    def helper(left, right):
        if left > right:
            return None

        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        root.left = helper(left, mid - 1)
        root.right = helper(mid + 1, right)
        return root

    return helper(0, len(nums) - 1)


def pascal_triangle(n: int) -> list[list[int]]:
    if n == 0:
        return []
    result: list[list[int]] = [[1]]
    for i in range(1, n):
        prev: list[int] = result[-1]
        tmp: list[int] = [1]
        for j in range(len(prev) - 1):
            tmp.append(prev[j] + prev[j + 1])
        tmp.append(1)
        result.append(tmp)
    return result


def single_number(nums: list[int]) -> int | None:
    res: int = 0
    for n in nums:
        res ^= n
    return res


def pascal_triangle_2(row_index: int) -> list[int]:
    result: list[int] = [1]
    if row_index == 0:
        return result
    for r in range(1, row_index + 1):
        result.append(result[-1] * (row_index - r + 1) // r)
    return result


def is_valid_palindrome(s: str) -> bool:
    s = "".join(filter(str.isalnum, s)).lower()
    return s == s[::-1]


def max_profit_buy_sell(prices: list[int]) -> int:
    min_price = float("inf")
    max_profit = 0
    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit


def has_cycle(head: Optional[ListNode]) -> bool:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def preorder_traversal(root: Optional[TreeNode]) -> list[int]:
    res = []

    def func(r):
        if r is None:
            return
        res.append(r.val)
        func(r.left)
        func(r.right)

    func(root)
    return res


def postorder_traversal(root: Optional[TreeNode]) -> list[int]:
    res = []

    def func(r):
        if r is None:
            return
        func(r.left)
        func(r.right)
        res.append(r.val)

    func(root)
    return res


def majority_element(nums: list[int]) -> Any:
    # Algorithme de Boyer-Moore Voting
    candidate = None
    count = 0

    for num in nums:
        if count == 0:
            candidate = num
            count = 1
        elif num == candidate:
            count += 1
        else:
            count -= 1

    return candidate


def roman_to_int(s: str) -> Any | None:
    if s == "":
        return None

    value: dict = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    res = value[s[-1]]

    for i in range(len(s) - 2, -1, -1):
        if value[s[i]] < value[s[i + 1]]:
            res -= value[s[i]]
        else:
            res += value[s[i]]
    return res


def transform_array(nums: list[int]) -> list[int]:
    for i in range(len(nums)):
        nums[i] = nums[i] % 2
    zeros = nums.count(0)
    for i in range(len(nums)):
        nums[i] = 0 if i < zeros else 1
    return nums


def get_intersection(h_a: ListNode, h_b: ListNode) -> Optional[ListNode]:
    if not h_a or not h_b:
        return None
    a, b = h_a, h_b
    while a is not b:
        a = a.next if a else h_a
        b = b.next if b else h_b
    return a


def is_cousins(root: Optional[TreeNode], x: int, y: int) -> bool:
    first = {}
    second = {}

    def dfs(node, parent, depth):
        if not node:
            return
        if node.val == x:
            first["parent"] = parent
            first["depth"] = depth
        if node.val == y:
            second["parent"] = parent
            second["depth"] = depth
        dfs(node.left, node, depth + 1)
        dfs(node.right, node, depth + 1)

    dfs(root, None, 0)
    if first["parent"] is not second["parent"] and first["depth"] == second["depth"]:
        return True
    return False


def third_max(nums: list[int]) -> int:
    tmp_set = set(nums)
    res = max(tmp_set)
    tmp_set.remove(res)
    i = 1
    while i < 3:
        if len(tmp_set) == 0:
            break
        res = max(tmp_set)
        tmp_set.remove(res)
        i += 1
    if i == 3:
        return res
    return max(nums)


def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    res = []
    occ1 = {}
    for n1 in nums1:
        occ1[n1] = occ1.get(n1, 0) + 1
    occ2 = {}
    for n2 in nums2:
        occ2[n2] = occ2.get(n2, 0) + 1
    for key in occ1:
        min_occ = min(occ1[key], occ2.get(key, 0))
        new = [key for _ in range(min_occ)]
        res += new
    return res


def move_zeroes(nums: list[int]) -> None:
    last_non_zero = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[i], nums[last_non_zero] = nums[last_non_zero], nums[i]
            last_non_zero += 1


def get_minimum_difference(root: Optional[TreeNode]) -> float:
    prev = None
    min_diff = float("inf")

    def in_order(node):
        if not node:
            return
        nonlocal prev, min_diff
        in_order(node.left)

        if prev is not None:
            min_diff = min(min_diff, abs(node.val - prev))
        prev = node.val

        in_order(node.right)

    in_order(root)
    return min_diff


def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    res = 0

    def post_order(node):
        nonlocal res
        if not node:
            return 0
        left = post_order(node.left)
        right = post_order(node.right)

        res = max(res, left + right)
        return max(left, right) + 1

    post_order(root)
    return res


def build_array(nums: list[int]) -> list[int]:
    ans: list[int] = nums[:]
    for i in range(len(nums)):
        num = nums[i]
        ans[i] = nums[num]

    return ans


def find_target(root: Optional[TreeNode], k: int) -> bool:
    seen = set()

    def dfs(node):
        if not node:
            return False
        if k - node.val in seen:
            return True
        seen.add(node.val)
        return dfs(node.left) or dfs(node.right)

    return dfs(root)


def find_tilt(root: Optional[TreeNode]) -> int:
    res: int = 0

    def postorder(node):
        nonlocal res
        if not node:
            return 0
        left: int = postorder(node.left)
        right: int = postorder(node.right)
        node_val: int = node.val
        node.val = abs(left - right)
        res += node.val
        return node_val + left + right

    postorder(root)
    return res


class KthLargest:

    def __init__(self, k: int, nums: list[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)

        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)

        while len(self.heap) > self.k:
            heapq.heappop(self.heap)

        return self.heap[0]


# 206. Reverse Linked List
def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    before = None
    current = head
    while current:
        if current.next is None:
            head = current
        after = current.next
        current.next = before
        before = current
        current = after
    return head


# 1984. Minimum Difference Between Highest and Lowest of K Scores
def minimum_difference(nums: list[int], k: int) -> int:
    min_diff = float("inf")
    nums.sort()
    if len(nums) == 1:
        return 0
    for i in range(len(nums) - k + 1):
        diff = nums[i + k - 1] - nums[i]
        min_diff = min_diff if min_diff < diff else diff
    return min_diff


def main():
    print(majority_element([1, 1, 3, 3, 3, 3, 3, 3, 4, 8, 9]))


if __name__ == "__main__":
    main()
