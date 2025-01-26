import numpy as np

# for ii in range(100, 0, -1):
#     print(ii)

DADAS_FAVOURITE_NUMBER = 6
MAMAS_FAVOURITE_NUMBER = 8
ZOES_FAVOURITE_NUMBER = 10
PHOEBES_FAVOURITE_NUMBER = 1

for counting_number in range(21):
    if counting_number == PHOEBES_FAVOURITE_NUMBER:
        print(counting_number, "\tHello Phoebe! Googoo gaga <I'm a baby> :D")
    if counting_number == ZOES_FAVOURITE_NUMBER:
        print(counting_number, "\tHello Zoe! I want some raisins :D")
    elif counting_number == MAMAS_FAVOURITE_NUMBER:
        print(counting_number, "\tHello Mama! Hope you feel better soon :D")
    elif counting_number == DADAS_FAVOURITE_NUMBER:
        print(counting_number, "\tHello Dada! I want some water and raisins, please :D")

    else:
        print(counting_number)


for ii in range (1000000+1, 0, -1):
    print(ii)
