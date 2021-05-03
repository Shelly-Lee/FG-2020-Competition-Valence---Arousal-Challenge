import matplotlib.pyplot as plt

v=[0.4870 , 0.4423, 0.4399, 0.4544, 0.4515, 0.4731]
a=[0.5484 , 0.5433, 0.5575, 0.5489, 0.5450, 0.5477, 0.5427, 0.5392]

plt.figure(figsize = (8, 4))
plt.plot(v,'ro-',label='valence')
plt.plot(a,'co-',label='arousal')
# plt.xlim((1, 8))
# plt.title('Epoch',fontsize=16, loc='center')
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('CCC Loss', fontsize=13)
# plt.xticks(rotation=90)

# plt.xticks(list(range(5)),['1','2','3','4','5','6','7','8'],fontsize=15)

plt.legend()
plt.grid(True)
plt.show()