import random

def guessNumber():
    number_is_gues = random.randint(1,9)
    while True:
        guess = int(input("Guess a number between 1 and 9: "))
        if guess == number_is_gues:
            print("Well guessed!")
            break
        else:
            print("Try again!")

guessNumber()