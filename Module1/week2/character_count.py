def character_count(word):
    character_statistic = {}
    for c in word:
        character_statistic[c] = character_statistic.get(c, 0) + 1
    return character_statistic

if __name__ == "__main__":
    print(character_count('Happiness'))
    print(character_count('smiles'))
