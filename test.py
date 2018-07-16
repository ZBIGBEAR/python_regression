from numpy import *
import regression as regress
import matplotlib.pyplot as plt
plt.switch_backend('agg')

if __name__=="__main__":
	k = input("input the k:")
	filename = "data/ex0.txt"
	xArr,yArr = regress.loadDataSet(filename)
	ws = regress.standRegres(xArr,yArr)
	#xMat是一个n*2的矩阵
	xMat = mat(xArr)
	#将xMat按照第二列排序，返回各个元素排序后的位置srtInd
	srtInd = xMat[:,1].argsort(0)
	#获取排序后的二维矩阵
	xSort = xMat[srtInd][:,0,:]
	#print("srtInd============")
	#print(srtInd)
	#print("xMat============")
	#print(xMat)
	#print("xSort===========")
	#print(xSort)
	yMat = mat(yArr)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	#print(xArr[:,1].flatten().A[0])
	#print(yArr.T[:,0].flatten().A[0])
	ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2,c='red')
	xCopy = xMat.copy()
	#yHat=xCopy*ws
	yHat = regress.lwlrTest(xArr,xArr,yArr,float(k))
	#print("=============yMat=====================")
	#print(yHat)
	#print(yHat[srtInd])
	#yHat[srtInd]表示将yHat按照srtInt为下标进行排序,yHat是一个1*n的矩阵，yHat[srtInd]是一个n*1的矩阵
	ax.plot(xSort[:,1],yHat[srtInd])
	plt.savefig(str(k)+'.jpg')
	#print("==============")
	#print(yMat)
	#print(yHat.T)
	cor = corrcoef(yHat.T,yMat)
	print(cor)
