def countNumberDigit(inputString):
    countNumber = 0
    countDigit = 0
    for char in inputString:
        if char.isalpha():
            countDigit += 1
        elif char.isdigit():
            countNumber += 1
    print(f"Letters: {countDigit}")
    print(f"Numbers: {countNumber}")


countNumberDigit("Python 3.2")
