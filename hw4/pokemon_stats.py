import math
import csv

def load_data(filepath):

    # Create a pokedex to hold the pokemon data, and a index to limit the reader.
    pokedex = []
    index = 0

    # Open the file and use csv.DictReader to read in rows.
    with open(filepath, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        # Read in each row as a dictionary with a list of (key, value) pairs.
        for row in reader:
            if (index < 20):
                pokemon = {"#": row['#'], "Name": row['Name'], "Type 1":row['Type 1'], "Type 2":row['Type 2'], "Total":row['Total'], "HP":row['HP'], "Attack":row['Attack'], "Defense":row['Defense'], "Sp. Atk":row['Sp. Atk'], "Sp. Def":row['Sp. Def'], "Speed":row['Speed']}
                # Add pokemon from specific row to the pokedex.
                pokedex.append(pokemon)
        # Return the pokedex
        return pokedex

def calculate_x_y(stats):

    # Import pokemon from function parameters.
    pokemon = stats
    
    # Calculate pokemon offensive stats.
    x = int(pokemon["Attack"]) + int(pokemon["Sp. Atk"]) + int(pokemon["Speed"])
    # Calculate pokemon defensive stats.
    y = int(pokemon["Defense"]) + int(pokemon["Sp. Def"]) + int(pokemon["HP"])

    # Return x and y as a touple value for later.
    return (x,y)

def hac(dataset):

    index = 0

    # Create an (m - 1) * 4
    matrix = [[None] * 4] * index

    # Find points that are closest together.
    # Give temp values, then replace below.
    tempDist = math.inf
    currentDist = 0

    item_1 = 0
    item_2 = 1

    for i in range(len(dataset)):
        for j in range(i + 1, len(dataset)):
            x1 = dataset[i][0] - dataset[j][0]
            x2 = dataset[i][1] - dataset[j][1]
            currentDist = math.sqrt(sum([x1 ** 2, x2 ** 2]))
            if tempDist > currentDist:
                tempDist = currentDist
                item_1 = i
                item_2 = j

    return (matrix)