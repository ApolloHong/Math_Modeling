import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 8), facecolor='w')
labels = ['90~100', '80~90', '70~80', '60~70', '60以下']
class1 = [10, 25, 35, 20, 10]
class2 = [25, 30, 20, 25]
colors = ['c', 'r', 'y', 'g', 'gray']

wedges1, texts1, autotexts1 = plt.pie(class1,
                                      autopct='%3.1f%%',
                                      radius=1,
                                      pctdistance=0.85,
                                      colors=colors,
                                      startangle=180,
                                      textprops={'color': 'w'},
                                      wedgeprops={'width': 0.3, 'edgecolor': 'w'}
                                      )
wedges2, texts2, autotexts2 = plt.pie(class2,
                                      autopct='%3.1f%%',
                                      radius=0.7,
                                      pctdistance=0.75,
                                      colors=colors,
                                      startangle=180,
                                      textprops={'color': 'w'},
                                      wedgeprops={'width': 0.3, 'edgecolor': 'w'}
                                      )

plt.legend(wedges1,
           labels,
           fontsize=12,
           title='成绩列表',
           loc='center right',
           bbox_to_anchor=(1.2, 0.6))

plt.text(0.7, 0.8, '甲班', fontsize=15, c='black')
plt.text(0.1, 0.1, '乙班', fontsize=15, c='black')

plt.setp(autotexts1, size=15, weight='bold')
plt.setp(autotexts2, size=15, weight='bold')
plt.setp(texts1, size=15)

plt.title('某校某次考试两班成绩对比', fontsize=20, c='black')

plt.show()
