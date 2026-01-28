from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.dates as mdates


releases = ['Started MS (9/20/21)', 'Started PhD (9/19/22)', 'Qualifying Exam (11/3/23)', 'Senate Exam (12/1/25)', 'Anticipated Graduation (11/1/26)']
dates = ['2021-09-20', '2022-09-19', '2023-11-03', '2025-12-01', '2026-11-01']

dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]  # Convert strs to dates.
releases = [tuple(release.split('.')) for release in releases]  # Split by component.
dates, releases = zip(*sorted(zip(dates, releases)))  # Sort by increasing date.


levels = [1, 0.5, 1, 0.5, 1]


# The figure and the axes.
fig, ax = plt.subplots(figsize=(8, 1), layout="constrained")

# The vertical stems.
ax.vlines(dates, 0, levels, color="tab:blue")
# The baseline.
ax.axhline(0, c="black")
# The markers on the baseline.
# meso_dates = [date for date, release in zip(dates, releases) if is_feature(release)]
micro_dates = [date for date, release in zip(dates, releases)]
ax.plot(micro_dates, np.zeros_like(micro_dates), "ko", mfc="tab:blue")
# ax.plot(meso_dates, np.zeros_like(meso_dates), "ko", mfc="tab:red")

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

# ax.margins(y=0.1)
# plt.savefig('phd_timeline.png', dpi=300)
plt.show()