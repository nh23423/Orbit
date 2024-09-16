
city_list0 = [

    'Vienna',
    'Tokyo',
    'London',
    'Birmingham',
    'Los Angeles',
    'Mapai',
    'Seoul',
    'Moscow',
    'Buenos Aires',
    'Shanghai',
    'Sydney'

]

def city_dict():
    with open('world_cities.csv', 'r') as f:
         lines = f.readlines()

    header = lines[0]

    cities = {}


    for line in lines[1:]:
        line = line.split(',')

        try:
            cities[line[1]] = [float(line[2]), float(line[3])]
        except:
            pass

    return cities
