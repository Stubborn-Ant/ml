# encoding=utf8
import filterspam
import mybayes
from numpy import *


def calcMostFreq(vocabList, fullTest):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullTest.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = filterspam.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = filterspam.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = mybayes.createVocabList(docList)

    # 去掉出现次数最多的那些词
    top30Word = calcMostFreq(vocabList, fullText)
    for pairW in top30Word:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = range(2 * minLen)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(mybayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = mybayes.trainNB0(array(trainMat), array(trainClasses))

    errorCount = 0
    for docIndex in testSet:
        wordVector = mybayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if mybayes.classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0v, p1v


def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]


if __name__ == '__main__':
    import feedparser

    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # vocabList, p0v, p1v = localWords(ny, sf)
    # print vocabList
    getTopWords(ny, sf)
