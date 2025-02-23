result = []
for number in range(1500,2501):
    if number % 7 == 0 and number % 5 == 0:
        result.append(number)
print(result)