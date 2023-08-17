from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt
import numpy as np


def plot_ex1():
    fig = plt.figure(1,[16,8]) #定义figure，（1）中的1是什么
    ax_cof = HostAxes(fig, [0, 0, 0.9, 0.9])  #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1

    #parasite addtional axes, share x
    ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_load = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_cp = ParasiteAxes(ax_cof, sharex=ax_cof)
    ax_wear = ParasiteAxes(ax_cof, sharex=ax_cof)

    #append axes
    ax_cof.parasites.append(ax_temp)
    ax_cof.parasites.append(ax_load)
    ax_cof.parasites.append(ax_cp)
    ax_cof.parasites.append(ax_wear)

    #invisible right axis of ax_cof
    ax_cof.axis['right'].set_visible(False)
    ax_cof.axis['top'].set_visible(False)
    ax_temp.axis['right'].set_visible(True)
    ax_temp.axis['right'].major_ticklabels.set_visible(True)
    ax_temp.axis['right'].label.set_visible(True)

    #set label for axis
    ax_cof.set_ylabel('cof')
    ax_cof.set_xlabel('Distance (m)')
    ax_temp.set_ylabel('Temperature')
    ax_load.set_ylabel('load')
    ax_cp.set_ylabel('CP')
    ax_wear.set_ylabel('Wear')

    load_axisline = ax_load.get_grid_helper().new_fixed_axis
    cp_axisline = ax_cp.get_grid_helper().new_fixed_axis
    wear_axisline = ax_wear.get_grid_helper().new_fixed_axis

    ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(40,0))
    ax_cp.axis['right3'] = cp_axisline(loc='right', axes=ax_cp, offset=(80,0))
    ax_wear.axis['right4'] = wear_axisline(loc='right', axes=ax_wear, offset=(120,0))

    fig.add_axes(ax_cof)

    ''' #set limit of x, y
    ax_cof.set_xlim(0,2)
    ax_cof.set_ylim(0,3)
    '''

    curve_cof, = ax_cof.plot([0, 1, 2], [0, 1, 2], label="CoF", color='black')
    curve_temp, = ax_temp.plot([0, 1, 2], [0, 3, 2], label="Temp", color='red')
    curve_load, = ax_load.plot([0, 1, 2], [1, 2, 3], label="Load", color='green')
    curve_cp, = ax_cp.plot([0, 1, 2], [0, 40, 25], label="CP", color='pink')
    curve_wear, = ax_wear.plot([0, 1, 2], [25, 18, 9], label="Wear", color='blue')

    ax_temp.set_ylim(0,4)
    ax_load.set_ylim(0,4)
    ax_cp.set_ylim(0,50)
    ax_wear.set_ylim(0,30)

    ax_cof.legend()

    #轴名称，刻度值的颜色
    #ax_cof.axis['left'].label.set_color(ax_cof.get_color())
    ax_temp.axis['right'].label.set_color('red')
    ax_load.axis['right2'].label.set_color('green')
    ax_cp.axis['right3'].label.set_color('pink')
    ax_wear.axis['right4'].label.set_color('blue')

    ax_temp.axis['right'].major_ticks.set_color('red')
    ax_load.axis['right2'].major_ticks.set_color('green')
    ax_cp.axis['right3'].major_ticks.set_color('pink')
    ax_wear.axis['right4'].major_ticks.set_color('blue')

    ax_temp.axis['right'].major_ticklabels.set_color('red')
    ax_load.axis['right2'].major_ticklabels.set_color('green')
    ax_cp.axis['right3'].major_ticklabels.set_color('pink')
    ax_wear.axis['right4'].major_ticklabels.set_color('blue')

    ax_temp.axis['right'].line.set_color('red')
    ax_load.axis['right2'].line.set_color('green')
    ax_cp.axis['right3'].line.set_color('pink')
    ax_wear.axis['right4'].line.set_color('blue')

    plt.savefig('9.key metrics mapping.pdf', bbox_inches='tight', dpi=800)
    plt.show()


def plot_ex2():
    x = ['ATL', 'LAX', 'CLT', 'LAS', 'MSP', 'DTW', 'PHX', 'DCA', 'SLC', 'ORD', 'DFW', 'PHL', 'PDX', 'DEN', 'IAH', 'BOS',
         'SAN', 'BWI', 'MDW', 'IND']
    k_in = [49.160, 47.367, 26.858, 30.315, 16.552, 28.590, 23.905, 18.818, 28.735, 6.721, 10.315, 26.398, 38.575,
            7.646, 11.227, 8.864, 15.327, 19.120, 11.521, 19.618]
    k_out = [38.024, 19.974, 25.011, 22.050, 30.108, 18.327, 20.811, 28.464, 23.72, 8.470, 4.119, 10.000, 25.158, 7.851,
             10.450, 11.130, 15.441, 7.519, 20.819, 32.825]
    p = [0.0537, 0.0301, 0.0306, 0.0217, 0.0229, 0.0223, 0.0218, 0.0179, 0.0155, 0.0465, 0.0419, 0.0165, 0.0091, 0.0357,
         0.0232, 0.0200, 0.0129, 0.0143, 0.0113, 0.0064]
    K = [4.6844, 2.0296, 1.5858, 1.1347, 1.0706, 1.0442, 0.9764, 0.8447, 0.8141, 0.7066, 0.6041, 0.5990, 0.5808, 0.5534,
         0.5023, 0.3992, 0.3964, 0.3799, 0.3639, 0.3331]
    fig = plt.figure(1)  # 定义figure

    ax_k = HostAxes(fig, [0, 0, 0.9, 0.9])  # 用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1

    # parasite addtional axes, share x
    ax_p = ParasiteAxes(ax_k, sharex=ax_k)
    ax_K = ParasiteAxes(ax_k, sharex=ax_k)

    # append axes
    ax_k.parasites.append(ax_p)
    ax_k.parasites.append(ax_K)

    ax_k.set_ylabel('$K_i^{in}\;/\;K_i^{out}$')
    ax_k.axis['bottom'].major_ticklabels.set_rotation(45)
    ax_k.set_xlabel('Airport')
    ax_k.axis['bottom', 'left'].label.set_fontsize(12)  # 设置轴label的大小
    ax_k.axis['bottom'].major_ticklabels.set_pad(8)  # 设置x轴坐标刻度与x轴的距离，坐标轴刻度旋转会使label和坐标轴重合
    ax_k.axis['bottom'].label.set_pad(12)  # 设置x轴坐标刻度与x轴label的距离，label会和坐标轴刻度重合
    ax_k.axis[:].major_ticks.set_tick_out(True)  # 设置坐标轴上刻度突起的短线向外还是向内

    # invisible right axis of ax_k
    ax_k.axis['right'].set_visible(False)
    ax_k.axis['top'].set_visible(True)
    ax_p.axis['right'].set_visible(True)
    ax_p.axis['right'].major_ticklabels.set_visible(True)
    ax_p.axis['right'].label.set_visible(True)
    ax_p.axis['right'].major_ticks.set_tick_out(True)
    ax_p.set_ylabel('${p_i}$')
    ax_p.axis['right'].label.set_fontsize(13)
    ax_K.set_ylabel('${K_i}$')

    K_axisline = ax_K.get_grid_helper().new_fixed_axis

    ax_K.axis['right2'] = K_axisline(loc='right', axes=ax_K, offset=(60, 0))
    ax_K.axis['right2'].major_ticks.set_tick_out(True)
    ax_K.axis['right2'].label.set_fontsize(13)
    fig.add_axes(ax_k)

    curve_k1, = ax_k.plot(list(range(20)), k_in, marker='v', markersize=8, label="$K_i^{in}$", alpha=0.7)
    curve_k2, = ax_k.plot(list(range(20)), k_out, marker='^', markersize=8, label="$K_i^{out}$", alpha=0.7)
    curve_p, = ax_p.plot(list(range(20)), p, marker='P', markersize=8, label="${p_i}$", alpha=0.7)
    curve_K, = ax_K.plot(list(range(20)), K, marker='o', markersize=8, label="${K_i}$", alpha=0.7, linewidth=3)
    plt.xticks(list(range(20)), x)
    # ax_k.set_xticks(list(range(20)))
    # ax_k.set_xticklabels(x)
    ax_k.axis['bottom'].major_ticklabels.set_rotation(45)

    # ax_k.set_rotation(90)
    # plt.xticks(list(range(20)), x, rotation = 'vertical')

    ax_p.set_ylim(0, 0.06)
    ax_K.set_ylim(0, 5)

    ax_k.legend(labelspacing=0.4, fontsize=10)

    # 轴名称，刻度值的颜色

    ax_p.axis['right'].label.set_color(curve_p.get_color())  # 坐标轴label的颜色
    ax_K.axis['right2'].label.set_color(curve_K.get_color())

    ax_p.axis['right'].major_ticks.set_color(curve_p.get_color())  # 坐标轴刻度小突起的颜色
    ax_K.axis['right2'].major_ticks.set_color(curve_K.get_color())

    ax_p.axis['right'].major_ticklabels.set_color(curve_p.get_color())  # 坐标轴刻度值的颜色
    ax_K.axis['right2'].major_ticklabels.set_color(curve_K.get_color())

    ax_p.axis['right'].line.set_color(curve_p.get_color())  # 坐标轴线的颜色
    ax_K.axis['right2'].line.set_color(curve_K.get_color())
    plt.savefig('10.key metrics mapping.pdf', bbox_inches='tight', dpi=800)
    plt.show()

plot_ex1()