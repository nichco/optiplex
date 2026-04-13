import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# # filter the routes by distance
# selected_routes = {}
# lower = 1000
# upper = 6000
# for route, distance in routes:
#     if lower <= distance <= upper:
#         # selected_routes.append(route)
#         selected_routes[route] = distance

# # print num routes
# print(f'Number of selected routes between {lower} and {upper} km: {len(selected_routes)}')

ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.stock_img()
ax.coastlines(color='black', zorder=3)
# ax.add_feature(cfeature.LAND, facecolor='tab:gray', alpha=0.5, zorder=1)
# ax.add_feature(cfeature.OCEAN, facecolor='tab:blue', alpha=0.1, zorder=0)


# # extract latitudes and longitudes
# lats = [coord[0] for coord in bases.values()]
# lons = [coord[1] for coord in bases.values()]
# names = list(bases.keys())

# plt.scatter(lons, lats, color='tab:blue', edgecolor='black', 
#             s=30, transform=ccrs.PlateCarree(), zorder=5)



# # for route in routes:
# for route in selected_routes:
#     a = route[0]
#     b = route[1]
#     plt.plot([bases[a][1], bases[b][1]], [bases[a][0], bases[b][0]], 
#              color='tab:orange', linewidth=1, alpha=0.5, transform=ccrs.Geodetic())


# plt.savefig('us_air_force_bases_and_routes.png', dpi=400, bbox_inches='tight', transparent=True)
plt.show()
