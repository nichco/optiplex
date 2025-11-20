from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.dates as mdates


releases = ['CSSO', 'CO', 'ATC', 'BLISS', 'Enhanced CO', 'AL-ATC', 'MDOQSS',
            'Response-Surface Optimization', 'Decision-Based CO', 'Response Surface BLISS', 
            'MDOISS', 'Nonhierarchical Decomposition Algorithm', 'Bilevel Adaptive Weighted Sum Method', 'ASO', 
            'Response Surface CSSO', 'Compact ATC', '2025']
dates = [1996, 1997, 2003, 2003.5, 2008, 2006, 2005, 2004, 2002, 2000, 2005.5, 1993, 2008.3, 2008.6, 1996.5, 2017, 2025]

releases = [tuple(release.split('.')) for release in releases]  # Split by component.
dates, releases = zip(*sorted(zip(dates, releases)))  # Sort by increasing date.


# levels = [1] * len(dates)

levels = [-1, 1, -0.7, -0.4, -0.4, 0.7, 0.5, 0.2, 1.2, 1.5, 0.4, 0.2, 0.9, -0.2, -0.7, -0.7, 1]

# # create new levels that alternate between 1 and -1
# for i in range(len(levels)):
#     if i % 2 == 0:
#         levels[i] = 1
#     else:
#         levels[i] = -1


# The figure and the axes.
fig, ax = plt.subplots(figsize=(8, 2), layout="constrained")

# The vertical stems.
ax.vlines(dates, 0, levels, color="tab:blue")
# The baseline.
ax.axhline(0, c="black")
# The markers on the baseline.
micro_dates = [date for date, release in zip(dates, releases)]
ax.plot(micro_dates, np.zeros_like(micro_dates), "ko", mfc="tab:blue")

# Annotate the lines.
for date, level, release in zip(dates, levels, releases):
    version_str = '.'.join(release)
    ax.annotate(version_str, xy=(date, level),
                xytext=(-3, np.sign(level)*3), textcoords="offset points",
                verticalalignment="bottom" if level > 0 else "top",
                weight="normal",
                bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

ax.xaxis.set(major_locator=mdates.YearLocator(),
             major_formatter=mdates.DateFormatter("%Y"))

# Remove the y-axis and some spines.
ax.yaxis.set_visible(False)
ax.spines[["left", "top", "right"]].set_visible(False)

plt.savefig('distributed_timeline.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()