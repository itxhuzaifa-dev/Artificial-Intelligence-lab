numbers = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

evenCount = 0
oddCount = 0

for num in numbers:
    if num % 2 == 0:
        evenCount += 1
    else:
        oddCount += 1
print(f"Number of even numbers: {evenCount}")
print(f"Number of odd numbers: {oddCount}")
