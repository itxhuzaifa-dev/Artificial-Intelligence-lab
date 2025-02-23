def calToFaren(cal):
    faren = ((cal * 9)/5) + 32
    return faren

def farenToCal(faren):
    cal = ((faren-32)*5)/9
    return cal

calu = 60
calToFar = calToFaren(calu)
print(f"{calu}°C is {calToFar:.0f} in fahrenheit")

far = 45
farToCal = farenToCal(far)
print(f"{far}°F is {farToCal:.0f} in Celsius")