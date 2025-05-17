from typing import Optional

from easy import ListNode


def generate_parenthesis(n: int) -> list[str]:
    def backtrack(s, left, right):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + "(", left + 1, right)
        if right < left:
            backtrack(s + ")", left, right + 1)

    result = []
    backtrack("", 0, 0)
    return result


def sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = sort_list(head)
    right = sort_list(mid)

    return merge(left, right)


def merge(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy

    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next

    tail.next = l1 if l1 else l2
    return dummy.next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode] | None:
    dummy: ListNode = ListNode(0, head)
    fast: ListNode | None = dummy
    for i in range(n):
        fast = fast.next
    if fast is None:
        return None
    prev: ListNode = dummy
    while fast and fast.next:
        fast = fast.next
        prev = prev.next
    after: ListNode = prev.next
    prev.next = after.next
    after.next = None
    return dummy.next


def insertion_sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    prev, curr = head, head.next
    while curr:
        if curr.val >= prev.val:
            prev, curr = curr, curr.next
            continue
        tmp = dummy
        while tmp.next.val < curr.val:
            tmp = tmp.next

        prev.next = curr.next
        curr.next = tmp.next
        tmp.next = curr
        curr = prev.next
    return dummy.next


def max_sub_array(nums: list[int]) -> int:
    if len(nums) == 1:
        return nums[0]
    max_sum = nums[0]
    max_using_element = [nums[0]]
    for i in range(1, len(nums)):
        num = nums[i]
        current_max = max(num, num + max_using_element[i - 1])
        max_using_element.append(current_max)
        max_sum = max(current_max, max_sum)
    return max_sum


def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    res = []

    def dfs(start: int, path: list[int], remaining: int):
        if remaining == 0:
            res.append(path[:])
            return
        if remaining < 0:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            dfs(i, path, remaining - candidates[i])
            path.pop()

    dfs(0, [], target)
    return res


def num_tilings(n: int) -> int:
    mod = 10**9 + 7
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n == 2:
        return 2

    dp = [0] * (n + 1)
    dp[0], dp[1], dp[2] = 1, 1, 2

    for i in range(3, n + 1):
        dp[i] = (2 * dp[i - 1] + dp[i - 3]) % mod

    return dp[n]


def min_operations(nums: list[int]) -> int:
    min_op = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            continue
        if i + 2 >= len(nums):
            break
        for j in range(i, i + 3):
            num = nums[j]
            nums[j] = 0 if num == 1 else 1
        min_op += 1
    return min_op if set(nums) == {1} else -1


# 1992. Find All Groups of Farmland
def find_farmland(land: list[list[int]]) -> list[list[int]]:
    m, n = len(land), len(land[0])
    result = []

    for i in range(m):
        for j in range(n):
            if land[i][j] == 1:

                row, col = i, j
                while row + 1 < m and land[row + 1][j] == 1:
                    row += 1
                while col + 1 < n and land[i][col + 1] == 1:
                    col += 1
                for r in range(i, row + 1):
                    for c in range(j, col + 1):
                        land[r][c] = 0
                result.append([i, j, row, col])

    return result


def main(): ...


if __name__ == "__main__":
    main()
