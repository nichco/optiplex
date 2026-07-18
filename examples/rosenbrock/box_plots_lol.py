import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize']

# fig, ax = plt.subplots(1, 2, figsize=(5.3333, 2.5), constrained_layout=True)
fig, ax = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)

monolithic_timing_data_n100 = np.load('monolithic_time_data_n100_07162026.npz')
albcd_timing_data_n100_N2 = np.load('albcd_time_data_n100_N2_07162026.npz')
albcd_timing_data_n100_N5 = np.load('albcd_time_data_n100_N5_07162026.npz')
albcd_timing_data_n100_N10 = np.load('albcd_time_data_n100_N10_07162026.npz')
bp = ax[0].boxplot([monolithic_timing_data_n100['times'], albcd_timing_data_n100_N2['times'], albcd_timing_data_n100_N5['times'], albcd_timing_data_n100_N10['times']],
                   positions=[1, 2, 3, 4],
                   vert=True, 
                   tick_labels=['Monolithic', 'BCD2', 'BCD5', 'BCD10'], 
                   patch_artist=True, 
                   medianprops={'color': 'k'}
                   )
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i, box in enumerate(bp['boxes']): box.set_facecolor(colors[i])
plt.setp(ax[0].get_xticklabels(), rotation=30, ha='right')


monolithic_timing_data_n1000 = np.load('monolithic_time_data_n1000_07162026.npz')
albcd_timing_data_n1000_N2 = np.load('albcd_time_data_n1000_N2_07162026.npz')
albcd_timing_data_n1000_N5 = np.load('albcd_time_data_n1000_N5_07162026.npz')
albcd_timing_data_n1000_N10 = np.load('albcd_time_data_n1000_N10_07162026.npz')
bp = ax[1].boxplot([monolithic_timing_data_n1000['times'], albcd_timing_data_n1000_N2['times'], albcd_timing_data_n1000_N5['times'], albcd_timing_data_n1000_N10['times']],
                   positions=[1, 2, 3, 4],
                   vert=True,
                   tick_labels=['Monolithic', 'BCD2', 'BCD5', 'BCD10'],
                   patch_artist=True,
                   medianprops={'color': 'k'},
                   )
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i, box in enumerate(bp['boxes']): box.set_facecolor(colors[i])
# for box in bp['boxes']: box.set_facecolor('lightgray')
plt.setp(ax[1].get_xticklabels(), rotation=30, ha='right')

ax[0].set_ylabel('Time (s)')
ax[0].set_title('100d Rosenbrock')
ax[0].set_yscale('log')
ax[1].set_ylabel('Time (s)')
ax[1].set_title('1000d Rosenbrock')
ax[1].set_yscale('log')




# n = np.array([100, 400, 600, 800, 1000])
# monolithic_memory = np.array([3.37, 16.12, 33.01, 56.63, 86.96])
# distributed_memory_2_subp = np.array([2.91, 6.14, 10.40, 16.35, 23.95])
# distributed_memory_5_subp = np.array([2.81, 3.34, 4.05, 5.00, 6.25])
# distributed_memory_10_subp = np.array([2.98, 3.1, 3.3, 3.53, 3.85])
n = np.array([100, 600, 1000])
monolithic_memory = np.array([3.37, 33.01, 86.96])
distributed_memory_2_subp = np.array([2.91, 10.40, 23.95])
distributed_memory_5_subp = np.array([2.81, 4.05, 6.25])
distributed_memory_10_subp = np.array([2.98, 3.3, 3.85])

x = np.arange(len(n))  # positions for groups
width = 0.22  # width of each bar

ax[2].bar(x - 1.5*width, monolithic_memory, width, label='Monolithic',
       facecolor='tab:blue', edgecolor='black')
ax[2].bar(x - 0.5*width, distributed_memory_2_subp, width, label='BCD2',
       facecolor='tab:orange', edgecolor='black')
ax[2].bar(x + 0.5*width, distributed_memory_5_subp, width, label='BCD5',
       facecolor='tab:green', edgecolor='black')
ax[2].bar(x + 1.5*width, distributed_memory_10_subp, width, label='BCD10',
       facecolor='tab:red', edgecolor='black')

ax[2].set_xlabel('n')
ax[2].set_ylabel('Memory (MB)')
ax[2].set_xticks(x)
ax[2].set_xticklabels(n)
ax[2].legend()
ax[2].set_title('Rosenbrock memory scaling')


plt.savefig('box_plot.pdf', bbox_inches='tight')
plt.show()