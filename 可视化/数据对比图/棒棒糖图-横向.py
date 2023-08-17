import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame({'group': map(chr, range(65, 85)), 'values': np.random.uniform(size=20)})

my_dpi = 96
plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)

ordered_df = df.sort_values(by='values')
my_range = range(1, len(df.index) + 1)

my_color = np.where(ordered_df['group'] == 'B', 'orange', 'skyblue')
my_size = np.where(ordered_df['group'] == 'B', 70, 30)

plt.hlines(y=my_range, xmin=0, xmax=ordered_df['values'], color=my_color, alpha=0.4)
plt.scatter(ordered_df['values'], my_range, color=my_color, s=my_size, alpha=1)
plt.yticks(my_range, ordered_df['group'])
plt.title("What about the B group?", loc='left')
plt.xlabel('Value of the variable')
plt.ylabel('Group')
plt.show()
plt.gca()

### 2.2.2.横向哑铃图

value1 = np.random.uniform(size=20)
value2 = value1 + np.random.uniform(size=20) / 4
df = pd.DataFrame({'group': map(chr, range(65, 85)), 'value1': value1, 'value2': value2})

my_dpi = 96
plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)

ordered_df = df.sort_values(by='value1')
my_range = range(1, len(df.index) + 1)

plt.hlines(y=my_range, xmin=ordered_df['value1'], xmax=ordered_df['value2'], color='grey', alpha=0.4)
plt.scatter(ordered_df['value1'], my_range, color='skyblue', alpha=1, label='value1')
plt.scatter(ordered_df['value2'], my_range, color='green', alpha=0.4, label='value2')
plt.legend()
plt.yticks(my_range, ordered_df['group'])
plt.title("Comparison of the value 1 and the value 2", loc='left')
plt.xlabel('Value of the variables')
plt.ylabel('Group')
plt.show()
plt.gca()
