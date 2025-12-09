import math

xlims = (1e-2, 1)
ylims = (1e-2, 100)

xbins = 60
ybins = 60

with open('surface_pts.txt', 'w') as f:
  for xi in range(xbins):
    for yi in range(ybins):
      x = xlims[0] + xi*(xlims[1] - xlims[0])/xbins
      y = ylims[0] + yi*(ylims[1] - ylims[0])/ybins
      f.write(f'{x} {y}\n')
