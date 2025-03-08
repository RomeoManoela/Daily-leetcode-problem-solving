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


def main():
    print(generate_parenthesis(3))
    print(generate_parenthesis(1))


if __name__ == "__main__":
    main()
