from collections import Counter

c = Counter()

c[3]=300
c[4]=400

c_sorted = c.most_common()
print(c_sorted)

for i, _ in c_sorted:
    print(i)