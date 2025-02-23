def convert_lower_lines():
    line = []
    while True:
        word = input("Enter a line (blank line to terminate): ")
        if word == "":
            break
        else:
            line.append(word.lower())
    return line


lines = convert_lower_lines()
print("lower_lines:")
for line in lines:
    print(line)
