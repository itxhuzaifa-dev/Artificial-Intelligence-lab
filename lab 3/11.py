def generate_Metrix(row, col):
    matrix = []
    for i in range(row):
        row = []
        for j in range(col):
            row.append(i * j)
        matrix.append(row)
    return matrix


print(generate_Metrix(row=3, col=4))
