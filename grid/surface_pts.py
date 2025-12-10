import math

xlims = (-2, 0)
ylims = (-2, 2)

xbins = 60
ybins = 60

with open('surface_pts.txt', 'w') as f:
  for xi in range(xbins):
    for yi in range(ybins):
      x = xlims[0] + float(xi)*(xlims[1] - xlims[0])/xbins
      y = ylims[0] + float(yi)*(ylims[1] - ylims[0])/ybins
      f.write(f'{10**x} {10**y}\n')
