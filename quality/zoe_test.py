def zoe_numbers():

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

def zoe_letters():

    for ii in range(26):
        print(chr(ii+65))

zoe_letters()