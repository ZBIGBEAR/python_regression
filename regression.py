#用线性回归找到最佳拟合直线
from numpy import *

#从文件中加载数据
def loadDataSet(filename):
	fr = open(filename)
	lines = fr.readlines()
	numFeat=len(lines[0].strip().split('\t'))-1
	dataMat = []
	labelMat = []
	for line in lines:
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat

#求直线拟合系数
def standRegres(xArr,yArr):
	xMat = mat(xArr)
	#print(xMat)
	#print("==========")
	#print(xMat.T)
	yMat = mat(yArr).T
	xTx = xMat.T*xMat
	if linalg.det(xTx)==0.0:
		print("this matrix is singular,cannot do inverse")
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws

#局部加权线性回归
def lwlr(testPoint,xArr,yArr,k=1.0):
	xMat = mat(xArr)
	yMat = mat(yArr).T
	m = shape(xMat)[0]
	weights = mat(eye((m)))
	for j in range(m):
		diffMat = testPoint-xMat[j,:]
		weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
	xTx = xMat.T * (weights*xMat)
	if linalg.det(xTx)==0.0:
		pritn("this matrix is singular,cannot do inverse")
		return
	ws = xTx.I * (xMat.T * (weights * yMat ))
	return testPoint *ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
	m = shape(testArr)[0]
	yHat = zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat
