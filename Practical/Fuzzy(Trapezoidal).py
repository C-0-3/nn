import numpy as np

class Fuzzy:
    def __init__(self, a, b1, b2, c):
        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.c = c

    def trapezoidal(self, x):
        if x <= self.a or x >= self.c:
            return 0
        elif self.a < x <= self.b1:
            return (x - self.a) / (self.b1 - self.a)
        elif self.a < x < self.b2:
            return 1
        elif self.b2 <= x < self.c:
            return (self.c - x) / (self.c - self.b2)
        return 0

    def area(self): return (self.b2 - self.b1 + self.c - self.a) / 2
    def center(self): return (self.a + self.b1 + self.b2 + self.c) / 4

t_ba = Fuzzy(10, 15, 20, 25)   # temperature below average
t_l  = Fuzzy(0, 10, 15, 20)    # temperature low
p_ba = Fuzzy(1.0, 1.25, 1.5, 1.75)  # pressure below average
p_l  = Fuzzy(0.5, 1.0, 1.25, 1.5)   # pressure low
hp_mh = Fuzzy(3.0, 3.5, 4.5, 5.0)   # heating power medium-high
hp_h  = Fuzzy(4.5, 5.0, 5.5, 6.0)   # heating power high
vo_ml = Fuzzy(1.0, 1.5, 2.0, 2.5)   # valve opening medium-low
vo_s  = Fuzzy(0.5, 1.0, 1.25, 1.5)  # valve opening small

# Inputs
t = 17.5
p = 1.3

z1 = min(t_ba.trapezoidal(t), p_ba.trapezoidal(p))
z2 = min(t_l.trapezoidal(t), p_l.trapezoidal(p))
print("z1 =", z1, "\nz2 = ", z2)

def defuzzify(z1, z2, set1, set2):
    num = (z1 * set1.area() * set1.center()) + (z2 * set2.area() * set2.center())
    den = (z1 * set1.area()) + (z2 * set2.area())
    return num / den

c1 = defuzzify(z1, z2, hp_mh, hp_h)
c2 = defuzzify(z1, z2, vo_ml, vo_s)
print("c1 and c2 =", c1, c2)