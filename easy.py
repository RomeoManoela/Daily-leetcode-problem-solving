# check palindrome
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

print(is_palindrome(121))