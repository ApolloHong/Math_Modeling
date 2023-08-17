import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pandas import *
from sklearn.neighbors import KernelDensity

data = read_csv('filename')
countries = [x for x in np.unique(data.country)]
colors = ['#0000ff', '#3300cc', '#660099', '#990066', '#cc0033', '#ff0000']

gs = matplotlib.gridspec.GridSpec(len(countries), 1)
fig = plt.figure(figsize=(16, 9))

i = 0

ax_objs = []
for country in countries:
    country = countries[i]
    x = np.array(data[data.country == country].score)
    x_d = np.linspace(0, 1, 1000)

    kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
    kde.fit(x[:, None])

    logprob = kde.score_samples(x_d[:, None])

    # creating new axes object
    ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))

    # plotting the distribution
    ax_objs[-1].plot(x_d, np.exp(logprob), color="#f0f0f0", lw=1)
    ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1, color=colors[i])

    # setting uniform x and y lims
    ax_objs[-1].set_xlim(0, 1)
    ax_objs[-1].set_ylim(0, 2.5)

    # make background transparent
    rect = ax_objs[-1].patch
    rect.set_alpha(0)

    # remove borders, axis ticks, and labels
    ax_objs[-1].set_yticklabels([])

    if i == len(countries) - 1:
        ax_objs[-1].set_xlabel("Test Score", fontsize=16, fontweight="bold")
    else:
        ax_objs[-1].set_xticklabels([])

    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)

    adj_country = country.replace(" ", "\n")
    ax_objs[-1].text(-0.02, 0, adj_country, fontweight="bold", fontsize=14, ha="right")

    i += 1

gs.update(hspace=-0.7)

fig.text(0.07, 0.85, "Distribution of Aptitude Test Results from 18 – 24 year-olds", fontsize=20)

plt.tight_layout()
plt.show()
