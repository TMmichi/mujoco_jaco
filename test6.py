mg = 3.62453914
mr = -1.52448535

wg = 0.985254
wr = 0.0147460764

sg = 15.1350412
sr = 0.006737947

cg = wg/sg
cr = wr/sr

m = (cg*mg + cr*mr)/(cg+cr)
print('weight, grasping: ',wg, ', reaching: ',wr)
print('coefficient, grasping: ',cg,', reaching: ', cr)
print(m)
