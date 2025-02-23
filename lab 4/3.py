def binarySearch(input, value):
    left, right = 0, len(input) - 1
    while left <= right:
        mid = (left + right) // 2
        if input[mid] == value:
            return mid
        elif input[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# Example usage


sorted_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
target_value = 7

# Perform binary search


result = binarySearch(sorted_array, target_value)

if result != -1:
    print(f"Element {target_value} is present at index {result}.")
else:
    print(f"Element {target_value} is not present in the array.")
