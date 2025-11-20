from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.dates as mdates


releases = ['1990', 'CSSO', 'CO', 'ATC', 'BLISS', 'Enhanced CO', 'AL-ATC', 'MDOQSS',
            'Response-Surface Optimization', 'Decision-Based CO', 'Response Surface BLISS', 
            'MDOISS', 'Nonhierarchical Decomposition', 'Bilevel Adaptive Weighted Sum Method', 'ASO', 
            'Response Surface CSSO', 'Compact ATC', '2025']
dates = [1990, 1996, 1997, 2003, 2003.5, 2008, 2006, 2005, 2004, 2002, 2000, 2005.5, 1993, 2008.3, 2008.6, 1996.5, 2017, 2025]

releases = [tuple(release.split('.')) for release in releases]  # Split by component.
dates, releases = zip(*sorted(zip(dates, releases)))  # Sort by increasing date.

levels = [1, 1, 0.6, -0.7, -0.4, -0.4, 0.7, 0.5, -0.2, 1.2, 1.5, 0.4, 0.2, 0.9, -0.2, -0.7, -0.7, 1]


# The figure and the axes.
fig, ax = plt.subplots(figsize=(8, 2), layout="constrained")
# fig, ax = plt.subplots(figsize=(8, 2))

# Convert numeric year values (possibly with fractional year) to datetimes.
def _year_fraction_to_datetime(y: float) -> datetime:
    year = int(y)
    frac = y - year
    # Convert fractional part of year to days (approximate, ignores leap years).
    days = int(round(frac * 365))
    return datetime(year, 1, 1) + timedelta(days=days)

date_objs = [_year_fraction_to_datetime(d) for d in dates]

# The vertical stems and markers. Allow highlighting specific years in red.
highlight_years = {1990, 2025}  # change this set to highlight other years
# Draw the stems individually so we can color certain years differently.
for date_obj, level in zip(date_objs, levels):
    color = 'red' if date_obj.year in highlight_years else "tab:blue"
    ax.vlines(date_obj, 0, level, color=color)

# The baseline.
ax.axhline(0, c="black")

# The markers on the baseline: draw individually to allow per-point facecolor.
for date_obj in date_objs:
    face = 'red' if date_obj.year in highlight_years else 'tab:blue'
    ax.plot(date_obj, 0, marker='o', markersize=6,
            markeredgecolor='k', markerfacecolor=face)

# Annotate the lines.
for date_obj, level, release in zip(date_objs, levels, releases):
    version_str = '.'.join(release)
    ax.annotate(version_str, xy=(date_obj, level),
                xytext=(-3, np.sign(level)*3), textcoords="offset points",
                verticalalignment="bottom" if level > 0 else "top",
                weight="normal",
                bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

# Use proper major locator/formatter for dates.
# ax.xaxis.set_major_locator(mdates.YearLocator(3))  # show ticks every 3 years

# Remove the y-axis and some spines.  Also hide the bottom spine so the axis
# frame line doesn't appear; the timeline baseline drawn with ax.axhline(0)
# will still be visible. If you want to remove that baseline too, remove or
# comment out the ax.axhline(...) call above.
ax.yaxis.set_visible(False)
ax.spines[["left", "top", "right", "bottom"]].set_visible(False)

ax.xaxis.set_visible(False)

# ax.tick_params(axis='x', which='major', rotation=45, pad=14, labelsize=8)

plt.savefig('distributed_timeline.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()