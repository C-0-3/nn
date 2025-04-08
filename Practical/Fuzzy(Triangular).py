import numpy as np

class FuzzySet:
    def __init__(self, a, b, c): self.a, self.b, self.c = a, b, c
    def membership(self, x): 
        if x <= self.b: return max(0, (x - self.a) / (self.b - self.a))
        elif x <= self.c: return max(0, (self.c - x) / (self.c - self.b))
        return 0
    def area(self): return 0.5 * (self.c - self.b)

T_below_avg = FuzzySet(15, 30, 45)
T_low = FuzzySet(-5, 10, 25)
P_below_avg = FuzzySet(1.25, 2, 2.75)
P_low = FuzzySet(0.25, 1, 1.75)
HP_med_high = FuzzySet(3.25, 4, 4.75)
HP_high = FuzzySet(4.25, 5, 5.75)
VO_med_low = FuzzySet(1.25, 2, 2.75)
VO_small = FuzzySet(0.25, 1, 1.75)

T_in, P_in = 16.5, 1.3

z1 = min(T_below_avg.membership(T_in), P_below_avg.membership(P_in))
z2 = min(T_low.membership(T_in), P_low.membership(P_in))
prweint("z1 = ", z1, "\nz2 = ", z2)

def defuzzify(z1, z2, set1, set2):
    num = (z1 * set1.area() * set1.b) + (z2 * set2.area() * set2.b)
    den = (z1 * set1.area()) + (z2 * set2.area())
    return num / den

c1 = defuzzify(z1, z2, HP_med_high, HP_high)
c2 = defuzzify(z1, z2, VO_med_low, VO_small)
print("c1 and c2 = ", c1, c2)
