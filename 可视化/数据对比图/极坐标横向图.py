from math import pi

import matplotlib.pyplot as plt


def rbc_plot(raw_data, raw_name):  # imput: two lists of values and names
    l = len(raw_data)
    # sort the lists by value
    temp_dict = {}
    for i, x in enumerate(raw_data):
        temp_dict[raw_name[i]] = x

    temp_list = sorted(temp_dict.items(), key=lambda kv: (kv[1], kv[0]))
    data = [x[1] for x in temp_list]
    prop = [(x / max(data)) * 100 for x in data]
    name = [x[0] for x in temp_list]

    # plot
    fig, ax = plt.subplots(figsize=(1.5 * l, 1.5 * l))
    ax = plt.subplot(projection='polar')
    startangle = 90
    colors = ["blue", "black", "brown", "red", "yellow", "green", "orange", "beige", "turquoise", "pink"]
    xs = []
    ys = []
    for i, x in enumerate(prop):
        xs.append((x * pi * 1.5) / 100)
        ys.append(-0.2 + i * 5 / l)

    left = (startangle * pi * 2) / 360  # this is to control where the bar starts
    # plot bars and points at the end to make them round
    for i, x in enumerate(xs):
        ax.barh(ys[i], x, left=left, height=3.5 / l, color=colors[i])

    plt.ylim(-4, 4)
    for i, x in enumerate(name):
        plt.text(pi / 2 + xs[i] - 0.1, ys[i], data[i], color="white")
        plt.text(pi / 2, ys[i], name[i])

    plt.xticks([])
    plt.yticks([])
    ax.spines.clear()
    plt.show()


# example
rbc_plot([80, 70, 30, 60, 50, 40], ['A', 'B', 'C', 'D', 'E', 'F'])
