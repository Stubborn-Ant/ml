# encoding=utf8
'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''

from numpy import *


def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t'))  # get number of fields
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 遍历分类函数所有可能的输入 找到基于D的最佳单层决策树  stump 树桩
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    # 输入：样本特征矩阵，指定特征，阈值，是否反转类别
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray  # 返回类别向量


# 基于加权输入进行决策
def buildStump(dataArr, classLabels, D):  # 输入：样本特征矩阵，样本标签，样本权重
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T  # 行向量转置为列向量
    m, n = shape(dataMatrix)  # m行 n列
    numSteps = 10.0;
    bestStump = {};
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity inf为无穷大
    for i in range(n):  # loop over all dimensions 遍历每一个属性
        rangeMin = dataMatrix[:, i].min()  # 每一列中的最小值
        rangeMax = dataMatrix[:, i].max()  # 每一列中的最大值
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)  # threshVal为阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # call stump classify with i, j, lessThan
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # calc total error multiplied by D
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                #     i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst  # 返回决策树(使得错误率最小的最优属性,判别式,阈值)，最小错误率，及其预测的类别


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []  # 存储弱分类器数组
    m = shape(dataArr)[0]  # 存储样本个数
    D = mat(ones((m, 1)) / m)  # init D to all equal  存储每个样本的权重
    aggClassEst = mat(zeros((m, 1)))  # 存储每个样本的类别估计累计值,最终的值跟样本的标签一致,正负值代表两个类别
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        # print "D:", D.T
        alpha = float(
            0.5 * log((1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        # print "classEst: ", classEst.T
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = multiply(D, exp(expon))  # Calc New D for next iteration
        D = D / D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha * classEst  # 聚合后的ClassEst
        # print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # 将aggClassEst的实数转换为只包含0,1值的向量
        errorRate = aggErrors.sum() / m
        # print "total error: ", errorRate
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])  # call stump classify
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print aggClassEst
    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    # 第一个参数代表分类器预测强度 第二个参数类标签
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # cursor 光标
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(array(classLabels) == 1.0)  # 正例个数
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)  # 负例个数
    sortedIndicies = predStrengths.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:  # 从最小的值开始,最小的值为负数,即分类为阴性
        if classLabels[index] == 1.0:  # 实际为阳性
            delX = 0;
            delY = yStep;
        else:
            delX = xStep;  # 实际为阴性
            delY = 0;
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the Area Under the Curve is: ", ySum * xStep


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    datMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(currLine[i]))
        datMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return datMat, labelMat


if __name__ == '__main__':
    # datMat, classLabels = loadSimpData()

    # 测试单层决策树
    # D = mat(ones((5, 1)) / 5)  # 各个样本的初试权重
    # bestStump, minError, bestClasEst = buildStump(datMat, classLabels, D)
    # print bestStump
    # print minError
    # print bestClasEst

    # 测试基于单层决策树的自适应提升算法
    # weakClassArr, aggClassEst = adaBoostTrainDS(datMat, classLabels)
    # print weakClassArr
    # print aggClassEst

    # 进行分类
    # print adaClassify([0, 0], weakClassArr)
    # print adaClassify([[5, 5], [4, 4], [-1, -1]], weakClassArr)

    datMat, classLabels = loadDataSet('../data/horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(datMat, classLabels, 50)
    # testData, testLabelArr = loadDataSet('../data/horseColicTest2.txt')

    # 测试错误率
    # prediction = adaClassify(testData, weakClassArr)
    # print prediction
    # errArr = mat(ones((67, 1)))
    # errCount = errArr[prediction != mat(testLabelArr).T].sum()
    # print errCount

    # 训练错误率
    # prediction2 = adaClassify(datMat, weakClassArr)
    # print prediction2
    # errArr = mat(ones((299, 1)))
    # errCount = errArr[prediction2 != mat(classLabels).T].sum()
    # print errCount / 299

    # ROC
    plotROC(aggClassEst.T, classLabels)
