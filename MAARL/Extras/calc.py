import math
x0 = 192
y0 = 131
dist = 1.168/0.05
angle = 1.238

x = x0 + dist * math.cos(angle)
y = y0 + dist * math.sin(angle)

print(x,y)