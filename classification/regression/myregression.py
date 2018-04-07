# encoding=utf8
from numpy import *


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


#
#     判断矩阵是否可逆:
#         首先,可逆矩阵A一定是n阶方阵
#         其次,满足一下中的一条:
#             A的行列式不为0
#             A的秩等于n（满秩）  即非奇异
#             A的转置矩阵可逆
#             A的转置矩阵乘以A可逆
#             存在一个n阶方阵B使得AB或者BA=单位矩阵
#
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  # 线性代数库 linalg   .det()计算行列式
        print "This matrix is singular, cannot do inverse"  # 奇异矩阵,非满秩
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):  # next 2 lines create weights matrix
        diffMat = testPoint - xMat[j, :]  #
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


# 局部线性加权回归
def lwlrTest(testArr, xArr, yArr, k=1.0):  # loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def plotResult(xArr):
    yHat = lwlrTest(xArr, xArr, yArr, 0.005)  # 局部加权线性回归
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yHat).T[:, 0].flatten().A[0], s=2, c='red')
    plt.show()


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def testAbanlneAge():
    abX, abY = loadDataSet('abalone.txt')

    # 训练集上的误差
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print '训练集上的误差###################'
    print rssError(abY[0:99], yHat01.T)
    print rssError(abY[0:99], yHat1.T)
    print rssError(abY[0:99], yHat10.T)

    # 测试集上的误差
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print '测试集上的误差###################'
    print rssError(abY[100:199], yHat01.T)
    print rssError(abY[100:199], yHat1.T)
    print rssError(abY[100:199], yHat10.T)

    # 简单线性回归
    print '简单线性回归rssError###################'
    ws = standRegres(abX[0:99], abY[0:99])
    yHat = mat(abX[100:199]) * ws
    print rssError(abY[100:199], yHat.T.A)


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate(消除) X0 take mean off of Y
    # regularize X's

    # np.mean(matrix):对所有元素求均值   np.mean(matrix,0):压缩行，对各列求均值  np.mean(matrix,1):压缩列，对各行求均值
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):  # regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  # calc mean then subtract it off
    inVar = var(inMat, 0)  # calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat


# 前向逐步回归(贪心算法)
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # can also regularize ys but will get smaller coef

    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))  # testing code remove
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        # print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


from time import sleep
import json
import urllib2


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):  # [],[],个数,年代,产品编号,当时的价格
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
        myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print 'problem with item %d' % i


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


if __name__ == '__main__':
    # xArr, yArr = loadDataSet('ex0.txt')

    # 画出最佳拟合曲线
    # ws = standRegres(xArr, yArr)
    # print ws
    # xMat = mat(xArr)
    # yMat = mat(yArr)
    # yHat = xMat * ws
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])  # .A 转换为数组
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy * ws
    # ax.plot(xCopy[:, 1], yHat)
    # plt.show()

    # print '******************'
    # print corrcoef(yHat.T, yMat)

    # 测试局部加权线性回归
    # yHatlwlr = lwlrTest(xArr, xArr, yArr, 0.5)
    # print  yHatlwlr

    # 画出局部加权线性回归
    # plotResult(xArr)

    # 鲍鱼年龄预测
    # testAbanlneAge()

    # 岭回归测试
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    # ridgeWeights = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 6, 6]]
    import matplotlib.pyplot as plt
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.plot(ridgeWeights)  # plot(矩阵):以行号(包括0)为横坐标,每列的值为纵坐标绘图
    plt.show()

    # 前向逐步回归
    # abX, abY = loadDataSet('abalone.txt')
    # stageWeights = stageWise(abX, abY, 0.001, 5000)
    # print stageWeights[-1, :]
    #
    # xMat = mat(abX)
    # yMat = mat(abY).T
    # xMat = regularize(xMat)
    # yM = mean(yMat, 0)
    # yMat = yMat - yM
    # weights = standRegres(xMat, yMat.T)
    # print weights.T
    #
    # import matplotlib.pyplot as plt
    #
    # figure = plt.figure()
    # ax = figure.add_subplot(111)
    # ax.plot(stageWeights)  # plot(矩阵):以行号(包括0)为横坐标,每列的值为纵坐标绘图
    # plt.show()

    # 预测乐高玩具套装价格
    # lgX = []
    # lgY = []
    # setDataCollect(lgX, lgY)
