# encoding=utf8

import mybayes
from numpy import *


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('../data/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('../data/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = mybayes.createVocabList(docList)

    # 生成训练集合和测试集合
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))  # 生成 [0,len)之间的随机数
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(mybayes.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = mybayes.trainNB0(array(trainMat), array(trainClasses))

    errorCount = 0
    for docIndex in testSet:
        wordVector = mybayes.setOfWords2Vec(vocabList, docList[docIndex])
        if mybayes.classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)

if __name__ == '__main__':
    print 'test'
    spamTest()
