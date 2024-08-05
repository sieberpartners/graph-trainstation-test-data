# Creating the synthetic test legacy data
trainstation_legacy1 = {
    "StationID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "StationName": ["Gare du Nord", "St Pancras", "Hauptbahnhof", "Centrale", "Atocha", "Sants", "Zurich HB", "Amsterdam Centraal", "Wien Hauptbahnhof", "Gare de Lyon", "München Hbf", "Antwerp Central"],
    "Location": ["Paris, France", "London, UK", "Berlin, Germany", "Milan, Italy", "Madrid, Spain", "Barcelona, Spain", "Zurich, Switzerland", "Amsterdam, Netherlands", "Vienna, Austria", "Paris, France", "Munich, Germany", "Antwerp, Belgium"],
    "Platforms": [36, 15, 16, 24, 21, 14, 26, 15, 12, 13, 32, 14],
    "OpenedYear": [1864, 1868, 1871, 1931, 1851, 1975, 1847, 1889, 2012, 1900, 1849, 1905]
}

trainstation_legacy2 = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Name": ["Gare du Nord", "St Pancras", "Hauptbahnhof", "Centrale", "Atocha", "Sants", "Zurich HB", "Amsterdam Centraal", "Wien Hauptbahnhof", "Gare de Lyon", "München Hbf", "Antwerp Central"],
    "City": ["Paris", "London", "Berlin", "Milan", "Madrid", "Barcelona", "Zurich", "Amsterdam", "Vienna", "Paris", "Munich", "Antwerp"],
    "Country": ["France", "UK", "Germany", "Italy", "Spain", "Spain", "Switzerland", "Netherlands", "Austria", "France", "Germany", "Belgium"],
    "NumberOfPlatforms": [36, 15, 16, 24, 21, 14, 26, 15, 12, 13, 32, 14],
    "YearOpened": [1864, 1868, 1871, 1931, 1851, 1975, 1847, 1889, 2012, 1900, 1849, 1905]
}

trainstation_legacy3 = {
    "StationCode": ["FRPAR", "UKLON", "DEBER", "ITMIL", "ESMAD", "ESBCN", "CHZRH", "NLAMS", "ATVIE", "FRPAR2", "DEMUC", "BEBRU"],
    "FullName": ["GARE DU NORD", "ST PANCRAS", "HAUPTBAHNHOF", "CENTRALE", "ATOCHA", "SANTS", "ZURICH HB", "AMSTERDAM CENTRAAL", "WIEN HAUPTBAHNHOF", "GARE DE LYON", "MÜNCHEN HBF", "ANTWERP CENTRAL"],
    "Address": ["PARIS, FRANCE", "LONDON, UK", "BERLIN, GERMANY", "MILAN, ITALY", "MADRID, SPAIN", "BARCELONA, SPAIN", "ZURICH, SWITZERLAND", "AMSTERDAM, NETHERLANDS", "VIENNA, AUSTRIA", "PARIS, FRANCE", "MUNICH, GERMANY", "ANTWERP, BELGIUM"],
    "PlatformsCount": ['36', '15', '16', '24', '21', '14', '26', '15', '12', '13', '32', '14'],
    "Established": ['1864', '1868', '1871', '1931', '1851', '1975', '1847', '1889', '2012', '1900', '1849', '1905']
}