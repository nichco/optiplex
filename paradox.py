import matplotlib.pyplot as plt
import numpy as np


time = np.linspace(0, 1, 300)

k = 10  # steepness
t0 = 0.5  # midpoint
knowledge = 1 / (1 + np.exp(-k * (time - t0)))
knowledge = 100 * knowledge  # scale to 0–100%



freedom = np.exp(-5 * time)

# Normalize to 100%
knowledge = 100 * knowledge / np.max(knowledge)
freedom = 100 * freedom / np.max(freedom)


plt.figure(figsize=(4, 3))
plt.plot(time, knowledge, label='KNOWLEDGE ABOUT\nTHE OBJECT OF DESIGN', linewidth=3)
# plt.plot(time, freedom, label='DESIGN\nFREEDOM', linewidth=3)
plt.plot(time, freedom, label='DESIGN FREEDOM', linewidth=3)


plt.fill_between(time, knowledge, 0, alpha=0.2, color='tab:blue')
plt.fill_between(time, freedom, 0, alpha=0.2, color='tab:orange')


plt.title('PARADOX OF ENGINEERING DESIGN', fontsize=10, fontweight='bold', ha='center')
plt.xlabel('Time into design process')
# Remove the y-axis label; only show 0% and 100% tick marks to indicate scale.
plt.yticks([0, 100], ['0%', '100%'])

plt.xlim(0, 1)
plt.ylim(0, 100)


# plt.legend(loc='upper left')


# Replace the boxed legend with direct labels connected to each curve.
# This anchors a short line from the text to the curve so it's clear which
# label belongs to which line.
ax = plt.gca()
# Pick x positions (in data coordinates) where the labels will sit. Tune as needed.
k_x = 0.5
f_x = 0.1

# Interpolate y positions on each curve at the chosen x positions.
k_y = np.interp(k_x, time, knowledge)
f_y = np.interp(f_x, time, freedom)

ax.annotate('Knowledge about\nthe object of design', xy=(k_x, k_y),
			xytext=(k_x + 0, k_y - 30), textcoords='data', color='tab:blue',
			weight='bold', fontsize=9,
			bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8),
			arrowprops=dict(arrowstyle='-', color='tab:blue'))

ax.annotate('Design freedom', xy=(f_x, f_y),
			xytext=(f_x + 0, f_y + 15), textcoords='data', color='tab:orange',
			weight='bold', fontsize=9,
			bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5),
			arrowprops=dict(arrowstyle='-', color='tab:orange'))

# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# Remove x-axis ticks and replace the bottom axis line with a single
# right-pointing arrow to indicate forward time.
ax.set_xticks([])
# Hide the bottom spine so we don't get a second line under the arrow.
ax.spines['bottom'].set_visible(False)
# Draw an arrow using axes-fraction coordinates so its placement is stable
# regardless of the data limits. Adjust the y fraction (0.03) to move it up/down.
ax.annotate('', xy=(1, 0), xycoords='axes fraction', xytext=(0, 0),
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', linewidth=1.0, color='k', shrinkA=0, shrinkB=0, mutation_scale=28))

plt.grid(False)
plt.tight_layout()
plt.savefig('paradox_of_engineering_design.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
