import scipy as sp
linreg = sp.stats.linregress(y_true, y_pred)
r2 = linreg.rvalue**2
pyplot.figure(figsize=(10, 7))

# 스타일
plt.style.use('default')
plt.scatter(y_true,y_pred,color="green")

# 축 라벨 
plt.xlabel('Actual values',size=20)
plt.ylabel('Predicted values',size=20)

# 축 레이블
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

plt.plot(np.unique(y_true), np.poly1d(np.polyfit(y_true, y_pred, 1))(np.unique(y_true)),color="red")
plt.rcParams["figure.figsize"] = (7, 4)
plt.text(2.1,1.9,'R-squred = %0.2f' % r2,fontsize=20)
