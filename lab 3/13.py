def binaryNumberDivibleBySeven(input_String):
    numbersDivisibleBySeven = []
    binary_numbers = input_String.split(",")
    for binary in binary_numbers:
        decimalNumber = int(binary, 2)

        if decimalNumber % 5 == 0:
            numbersDivisibleBySeven.append(binary)
    # ",".join(numbersDivisibleBySeven)

    return numbersDivisibleBySeven


print(binaryNumberDivibleBySeven("0100,0011,1010,1001,1100,1001"))
