import csv

### read data csv
country_csv = []
with open('continents-according-to-our-world-in-data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        country_csv.append(row)

covid_csv = []
with open('total_cases.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        covid_csv.append(row)

### read each country or region belongs to which continent
region = dict()
for i in range(1, len(country_csv)):
    row = country_csv[i]
    region[row[0]] = row[-1]

### read the number of total cases of each country or region and of each continent
pairs = []
countries, numbers = covid_csv[0], covid_csv[-1]
num_continent = dict()
lst_continent = dict()
for country, continent in region.items():
    num_continent[continent] = 0
    lst_continent[continent] = []

for i in range(1, len(numbers)):
    country, number = countries[i], int(float(numbers[i]))
    if country in region:
        pairs.append((number, country))
        num_continent[region[country]] += number

pairs = sorted(pairs, reverse=True)
threshold = 18  # how many countries should appear in the final figure

for i in range(threshold):
    number, country = pairs[i]
    lst_continent[region[country]].append(pairs[i])
num_continent.pop('Antarctica')
lst_continent.pop('Antarctica')

### process data
f = open('figure.js', 'w')
f.write("var data = [\n")
flag1 = False
for name, val in num_continent.items():
    if flag1: f.write(",\n")
    f.write("  {\n")
    f.write("    name: \'{name}\', value: {val},".format(name=name, val=val))
    if lst_continent[name] == []:
        f.write(R" label: {position: 'outside', ")
        f.write("distance: {dis}".format(dis=31 if name == 'Africa' else 25))
        f.write(R"}, labelLine: {show: true},")
    f.write("\n    children: [")
    flag2 = False
    for val, name in lst_continent[name]:
        if flag2: f.write(", ")
        f.write("{ ")
        f.write("name: \'{name}\', value: {val}".format(name=name, val=val))
        f.write(" }")
        flag2 = True
    f.write("]\n  }")
    flag1 = True
f.write("""
];
option = {
  title: {
    text: "Total confirmed cases of COVID-19",
""")
f.write("    subtext: \'As of {date}\\nSources: John Hopkins University and Our World in Data\',".format(
    date=covid_csv[-1][0]))
f.write("""
    left: "center"
  },
  series: {
    type: 'sunburst',
    data: data,
    radius: [0, '75%'],
    label: {
      rotate: 'radial'
    }
  }
};
""")
