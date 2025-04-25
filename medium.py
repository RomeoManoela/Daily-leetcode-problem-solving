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


def sort_list(self, head: Optional[ListNode]) -> Optional[ListNode]:

    if not head or not head.next:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = self.sortList(head)
    right = self.sortList(mid)

    return merge_ll(left, right)


def merge_ll(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
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
        return
    prev: ListNode = dummy
    while fast and fast.next:
        fast = fast.next
        prev = prev.next
    after: ListNode = prev.next
    prev.next = after.next
    after.next = None
    return dummy.next


def main():
    print(generate_parenthesis(3))
    print(generate_parenthesis(1))


if __name__ == "__main__":
    main()
