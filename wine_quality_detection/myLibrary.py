import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.special as spsp
import scipy.optimize as spop
import matplotlib.pyplot as plt
from functools import reduce


def load(input):
    attributes = []
    labels = []

    with open(input, 'r') as f:
        for line in f:
            data = line.split(',')[:11]
            data = mcol(np.array([float(i) for i in data]))
            attributes.append(data)
            label = line.split(',')[-1].strip()
            labels.append(label)

    return np.hstack(attributes), np.array([int(i) for i in labels])


def extractLowQuality(data, label):
    return data[:, label == 0]


def extractHighQuality(data, label):
    return data[:, label == 1]


def mcol(v):
    return v.reshape(v.size, 1)


def vrow(v):
    return v.reshape(1, v.size)


def empMean(X):
    return mcol(X.mean(1))


def empCovariance(X):
    mu = empMean(X)
    cenX = X - mu
    return np.dot(cenX, cenX.T) / cenX.shape[1]


def featureNormalization(data):
    featuresMinMax = dict()
    for feature in range(data.shape[0]):
        vector = data[feature, :]
        maxValue = np.max(vector)
        minValue = np.min(vector)
        featuresMinMax[feature] = (minValue, maxValue)
    return featuresMinMax


def dataNormalization(data, featuresMinMax):
    normalizedData = np.zeros(data.shape)
    for feature in range(data.shape[0]):
        vector = np.array(data[feature, :])
        vector -= featuresMinMax[feature][0]
        vector /= (featuresMinMax[feature][1] - featuresMinMax[feature][0])
        normalizedData[feature, :] = vector
    return normalizedData


def eigenValues_Vectors(data):
    mu = data.mean(1)
    cenData = data - mcol(mu)
    coMatrix = np.dot(cenData, cenData.T) / cenData.shape[1]
    eigenValues, eigenVectors = np.linalg.eigh(coMatrix)
    return eigenValues, eigenVectors


def PCA(data, m=None):
    if m is None:
        m = data.shape[0]
    U = eigenValues_Vectors(data)[1]
    projMatrix = U[:, ::-1][:, :m]
    return projMatrix


def computeSB(data, label, numberOfClass):
    SB = 0
    mu = empMean(data)
    N = data.shape[1]
    for i in range(numberOfClass):
        dataI = data[:, label == i]
        muI = empMean(dataI)
        cenClass = muI - mu
        SBI = np.dot(cenClass, cenClass.T) * dataI.shape[1]
        SB += SBI
    SB /= N
    return SB


def computeSW(data, label, numberOfClass):
    SW = 0
    N = data.shape[1]
    for i in range(numberOfClass):
        dataI = data[:, label == i]
        coMatrix = empCovariance(dataI)
        SW += coMatrix * dataI.shape[1]
    SW /= N
    return SW


def LDA(data, label, numberOfClass, m=None):
    if m is None:
        m = data.shape[0]
    SB = computeSB(data, label, numberOfClass)
    SW = computeSW(data, label, numberOfClass)
    eigenVectors = spla.eigh(SB, SW)[1]
    W = eigenVectors[:, ::-1][:, 0:m]
    return W


def logpdf_GAU_ND(X, mu, C):
    M = X.shape[0]
    invC = np.linalg.inv(C)
    detC = np.linalg.slogdet(C)[1]
    const = -0.5*M*np.log(2*np.pi)
    const -= 0.5*detC
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1] - mu
        result = const - 0.5*np.dot(x.T, np.dot(invC, x))
        Y.append(result)
    return np.array(Y).ravel()


def multiVariateGaussian_model(trainingData, trainingLabels, numberOfClass):
    classDict = {}
    for i in range(numberOfClass):
        mu = empMean(trainingData[:, trainingLabels == i])
        C = empCovariance(trainingData[:, trainingLabels == i])
        classDict[i] = (mu, C)

    return classDict


def naiveBayesGaussian_model(trainingData, trainingLabels, numberOfClass):
    classDict = {}
    for i in range(numberOfClass):
        mu = empMean(trainingData[:, trainingLabels == i])
        C = empCovariance(trainingData[:, trainingLabels == i])
        C = np.eye(C.shape[0]) * C
        classDict[i] = (mu, C)

    return classDict


def tiedCovGaussian_model(trainingData, trainingLabels, numberOfClass):
    classDict = {}
    for i in range(numberOfClass):
        mu = empMean(trainingData[:, trainingLabels == i])
        C = computeSW(trainingData, trainingLabels, numberOfClass)
        classDict[i] = (mu, C)

    return classDict


def tiedNaiveBayesGaussian_model(trainingData, trainingLabels, numberOfClass):
    classDict = {}
    for i in range(numberOfClass):
        mu = empMean(trainingData[:, trainingLabels == i])
        C = computeSW(trainingData, trainingLabels, numberOfClass)
        C = np.eye(C.shape[0]) * C
        classDict[i] = (mu, C)

    return classDict


def Gaussian_classification(classDict, newData, numberOfClass, classPriors=None):
    scoreJoint = np.zeros((numberOfClass, newData.shape[1]))
    if classPriors is None:
        classPriors = [1.0/numberOfClass]*numberOfClass

    for i in range(numberOfClass):
        scoreJoint[i, :] = logpdf_GAU_ND(
            newData, classDict[i][0], classDict[i][1]) + np.log(classPriors[i])
        scoreMarginal = spsp.logsumexp(scoreJoint, axis=0)
        scorePosteriorProbability = np.exp(
            scoreJoint - scoreMarginal)

    return scorePosteriorProbability


def logRegObj_wrap_binary(data, label, la):
    def logisticRegressionObjective(v):
        w = mcol(v[0:-1])
        b = v[-1]
        Z = 2*label - 1
        result = 0
        for i in range(data.shape[1]):
            temp = np.dot(w.T, data[:, i:i+1])
            temp = -1 * Z[i] * (temp + b)
            result += np.logaddexp(0, temp)
        result /= data.shape[1]
        result += 0.5 * la * (np.linalg.norm(w))**2
        return result.ravel()
    return logisticRegressionObjective


def logisticRegression_binary_model(trainingData, trainingLabels, la):
    X0 = np.zeros(trainingData.shape[0]+1)
    RES = spop.fmin_l_bfgs_b(logRegObj_wrap_binary(
        trainingData, trainingLabels, la), X0, approx_grad=True, maxfun=15000, factr=10000000.0)
    w = mcol(RES[0][0:-1])
    b = RES[0][-1]
    # j = RES[1]
    return w, b


def logisticRegression_binary_classification(newData, w, b, threshold=0):
    score = np.zeros(newData.shape[1])
    for i in range(newData.shape[1]):
        # temp = np.dot(w.T, newData[:, i:i+1]) + b
        # score[i] = 1 if temp > threshold else 0
        score[i] = np.dot(w.T, newData[:, i:i+1]) + b
    return score


def LinearSVM_model(data, labels, C, K=1):
    extendedData = np.vstack([data, np.ones((1, data.shape[1]))*K])
    Z = np.zeros(labels.shape)
    Z[labels == 1] = 1
    Z[labels == 0] = -1
    G = np.dot(extendedData.T, extendedData)
    H = mcol(Z) * vrow(Z) * G

    def JDual(alpha):
        temp = np.dot(H, mcol(alpha))
        Jhat = -1 * 0.5 * np.dot(vrow(alpha), temp).ravel()
        Jhat += alpha.sum()
        grad = np.ones(alpha.size) - temp.ravel()
        return Jhat, grad

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -1*loss, -1*grad

    alphaStar, _x, _y = spop.fmin_l_bfgs_b(LDual, np.zeros(data.shape[1]),
                                           bounds=[(0, C)] * data.shape[1], factr=1.0, maxiter=100000, maxfun=100000)

    wStar = np.dot(extendedData, mcol(alphaStar) * mcol(Z))
    return wStar


def LinearSVM_classification(newData, wStar, K=1):
    extendedData = np.vstack([newData, np.ones((1, newData.shape[1]))*K])
    score = np.dot(vrow(wStar), extendedData)
    # score[score >= 0] = 1
    # score[score < 0] = 0
    return score.ravel()


def PolynomialKernel(x1, x2, c, d, K=1):
    result = np.dot(x1.T, x2) + c
    result **= d
    result += K**2
    return result


def SVM_PolynomialKernel_model(data, labels, c, d, C, K=1):
    Z = np.zeros(labels.shape)
    Z[labels == 1] = 1
    Z[labels == 0] = -1
    Khat = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            Khat[i, j] = PolynomialKernel(data[:, i], data[:, j], c, d, K)
    H = mcol(Z) * vrow(Z) * Khat

    def JDual(alpha):
        temp = np.dot(H, mcol(alpha))
        Jhat = -1 * 0.5 * np.dot(vrow(alpha), temp).ravel()
        Jhat += alpha.sum()
        grad = np.ones(alpha.size) - temp.ravel()
        return Jhat, grad

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -1*loss, -1*grad

    alphaStar, _x, _y = spop.fmin_l_bfgs_b(LDual, np.zeros(data.shape[1]),
                                           bounds=[(0, C)] * data.shape[1], factr=1.0, maxiter=100000, maxfun=100000)

    return alphaStar


def SVM_PolynomialKernel_classification(trainingData, trainingLabels, testData, alphaStar, c, d, K=1):
    Z = np.zeros(trainingLabels.shape)
    Z[trainingLabels == 1] = 1
    Z[trainingLabels == 0] = -1
    score = np.zeros((testData.shape[1], ))

    for l in range(testData.shape[1]):
        for i in range(trainingData.shape[1]):
            if alphaStar[i] > 0:
                score[l] += Z[i] * alphaStar[i] * \
                    PolynomialKernel(
                        trainingData[:, i], testData[:, l], c, d, K)
    # score[score >= 0] = 1
    # score[score < 0] = 0
    return score


def RadialBasisKernel(x1, x2, gamma, K=1):
    result = x1 - x2
    result = -1 * np.linalg.norm(result) ** 2
    result = np.exp(gamma * result)
    result += K**2
    return result


def SVM_RadialBasisKernel_model(data, labels, gamma, C, K=1):
    Z = np.zeros(labels.shape)
    Z[labels == 1] = 1
    Z[labels == 0] = -1
    Khat = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            Khat[i, j] = RadialBasisKernel(data[:, i], data[:, j], gamma, K)
    H = mcol(Z) * vrow(Z) * Khat

    def JDual(alpha):
        temp = np.dot(H, mcol(alpha))
        Jhat = -1 * 0.5 * np.dot(vrow(alpha), temp).ravel()
        Jhat += alpha.sum()
        grad = np.ones(alpha.size) - temp.ravel()
        return Jhat, grad

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -1*loss, -1*grad

    alphaStar, _x, _y = spop.fmin_l_bfgs_b(LDual, np.zeros(data.shape[1]),
                                           bounds=[(0, C)] * data.shape[1], factr=1.0, maxiter=100000, maxfun=100000)

    return alphaStar


def SVM_RadialBasisKernel_classification(trainingData, trainingLabels, testData, alphaStar, gamma, K=1):
    Z = np.zeros(trainingLabels.shape)
    Z[trainingLabels == 1] = 1
    Z[trainingLabels == 0] = -1
    score = np.zeros((testData.shape[1], ))

    for l in range(testData.shape[1]):
        for i in range(trainingData.shape[1]):
            if alphaStar[i] > 0:
                score[l] += Z[i] * alphaStar[i] * \
                    RadialBasisKernel(
                        trainingData[:, i], testData[:, l], gamma, K)
    # score[score >= 0] = 1
    # score[score < 0] = 0
    return score


def logpdf_GMM(X, gmm):
    S = np.zeros((len(gmm), X.shape[1]))
    for g in range(len(gmm)):
        Y = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])
        S[g, :] = Y
        S[g, :] += np.log(gmm[g][0])
    return S


def GMM_EM(X, gmm, delta, psi, model=None):
    oldAverageLogLikelihood = None
    newAverageLogLikelihood = None
    while oldAverageLogLikelihood is None or newAverageLogLikelihood - oldAverageLogLikelihood > delta:
        oldAverageLogLikelihood = newAverageLogLikelihood
        joinDensities = logpdf_GMM(X, gmm)
        marginalDensities = spsp.logsumexp(joinDensities, axis=0)
        newAverageLogLikelihood = marginalDensities.sum() / X.shape[1]
        posteriorProbabilities = np.exp(joinDensities - marginalDensities)
        newGmm = []
        # used for tied model
        sigma = 0
        for g in range(len(gmm)):
            gamma = posteriorProbabilities[g, :]
            Zg = gamma.sum()
            Fg = (vrow(gamma) * X).sum(1)
            Sg = np.dot(X, (vrow(gamma) * X).T)
            w = Zg / X.shape[1]
            mu = mcol(Fg / Zg)
            C = Sg / Zg - np.dot(mu, mu.T)
            if model == 'diagonal':
                C = C * np.eye(C.shape[0])
            if model == 'tied':
                sigma += Zg * C
            if model != 'tied':
                U, s, _ = np.linalg.svd(C)
                s[s < psi] = psi
                C = np.dot(U, mcol(s)*U.T)
            newGmm.append((w, mu, C))
        if model == 'tied':
            sigma = sigma/X.shape[1]
            U, s, _ = np.linalg.svd(sigma)
            s[s < psi] = psi
            sigma = np.dot(U, mcol(s)*U.T)
            for g in range(len(newGmm)):
                newGmm[g] = (newGmm[g][0], newGmm[g][1], sigma)
        gmm = newGmm
        # print(newAverageLogLikelihood)
    return gmm


def GMM_LBG(X, alpha, delta, psi, numberOfComponents, model=None):
    mu = empMean(X)
    C = empCovariance(X)
    initialGMM = [(1.0, mu, C)]
    loopCondition = True
    while loopCondition:
        temp = []
        for i in initialGMM:
            U, s, Vh = np.linalg.svd(i[2])
            d = U[:, 0:1] * s[0]**0.5 * alpha
            w = i[0]/2
            sigma = i[2]
            mu1 = i[1] + d
            mu2 = i[1] - d
            temp.append((w, mu1, sigma))
            temp.append((w, mu2, sigma))
        finalGMM = GMM_EM(X, temp, delta, psi, model)
        if (len(finalGMM) >= numberOfComponents):
            loopCondition = False
        initialGMM = finalGMM

    return finalGMM


def GMM_model(trainingData, trainingLabels, numberOfClass, numberOfComponents, alpha, delta, psi, model=None):
    classDict = {}
    for i in range(numberOfClass):
        gmm = GMM_LBG(trainingData[:, trainingLabels == i],
                      alpha, delta, psi, numberOfComponents, model)
        classDict[i] = gmm
    return classDict


def GMM_classification(classDict, newData, numberOfClass):
    marginalDensities = np.zeros((numberOfClass, newData.shape[1]))
    for i in range(numberOfClass):
        joinDensities = logpdf_GMM(newData, classDict[i])
        marginalDensities[i, :] = spsp.logsumexp(joinDensities, axis=0)
    return marginalDensities


def shuffleData(data, label, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(data.shape[1])
    index = idx[:]
    DTR = data[:, index]
    LTR = label[index]
    return (DTR, LTR)


def computeError(score, testLabels):
    accuracy = np.array([score == testLabels]).sum() / testLabels.ravel().size
    error = (1 - accuracy) * 100
    return error


def confusionMatrix(numberOfClass, score, testLabels):
    confusion = np.zeros((numberOfClass, numberOfClass))
    for i in range(score.size):
        classIndex = testLabels[i]
        predictedIndex = score[i]
        confusion[predictedIndex, classIndex] += 1
    return confusion


def binaryOptimalBayesDecision(pi, Cfn, Cfp, logLikelihoodRatio, threshold=None):
    if threshold == None:
        threshold = -1 * np.log((pi*Cfn) / ((1-pi)*Cfp))
    score = [1 if logLikelihoodRatio[i] >
             threshold else 0 for i in range(logLikelihoodRatio.size)]
    return np.array(score)


def binaryBayesRisk(pi, Cfn, Cfp, LLR, labels, threshold=None):
    score = binaryOptimalBayesDecision(pi, Cfn, Cfp, LLR, threshold)
    confusion = confusionMatrix(2, score, labels)
    FN = confusion[0][1]
    TP = confusion[1][1]
    FP = confusion[1][0]
    TN = confusion[0][0]
    FNR = FN / (FN+TP)
    FPR = FP / (FP+TN)
    DCF = pi*Cfn*FNR + (1-pi)*Cfp*FPR
    normalizeDCF = DCF / min(pi*Cfn, (1-pi)*Cfp)
    return normalizeDCF, FNR, FPR


def binaryMinDCF(pi, Cfn, Cfp, LLR, labels):
    normalizeDCFs = []
    for i in LLR:
        normalizeDCF = binaryBayesRisk(pi, Cfn, Cfp, LLR, labels, i)[0]
        normalizeDCFs.append(normalizeDCF)
    min_normalizeDCF = reduce(lambda a, b: min(a, b), normalizeDCFs)
    return min_normalizeDCF


def binaryDrawROC(pi, Cfn, Cfp, LLR, labels):
    TPRs = []
    FPRs = []
    for i in LLR:
        normalizeDCF, FNR, FPR = binaryBayesRisk(pi, Cfn, Cfp, LLR, labels, i)
        FPRs.append(FPR)
        TPRs.append(1-FNR)
    plt.figure()
    plt.plot(np.sort(np.array(FPRs)), np.sort(np.array(TPRs)))
    plt.show()


def drawBayesErrorPlot(LLR, label):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    normalizeDCFs = []
    min_normalizeDCFs = []
    for i in effPriorLogOdds:
        pTilda = 1 / (1 + np.exp(-1*i))
        normalizeDCF = binaryBayesRisk(pTilda, 1, 1, LLR, label)[0]
        normalizeDCFs.append(normalizeDCF)
        min_normalizeDCF = binaryMinDCF(pTilda, 1, 1, LLR, label)
        min_normalizeDCFs.append(min_normalizeDCF)
    plt.plot(effPriorLogOdds, np.array(normalizeDCFs), label='DCF', color='r')
    plt.plot(effPriorLogOdds, np.array(
        min_normalizeDCFs), label='min DCF', color='b')
    plt.ylim([0, 0.8])
    plt.xlim([-3, 3])
    plt.show()


def KFold_Gaussian(D, L, numberOfClass, K, pi, Cfn, Cfp, model, mPCA=None):
    (data, label) = shuffleData(D, L)
    idx = range(data.shape[1])
    interval = int(data.shape[1] / float(K))
    foldedLLR = np.zeros((K, interval))

    for i in range(K):
        startIndex = i * interval
        endIndex = startIndex + interval

        idxTest = idx[startIndex:endIndex]
        idxTrain = np.delete(idx, idx[startIndex:endIndex])

        trainData = data[:, idxTrain]
        testData = data[:, idxTest]
        trainLabel = label[idxTrain]

        featuresMinMax = featureNormalization(trainData)
        trainData = dataNormalization(trainData, featuresMinMax)
        testData = dataNormalization(testData, featuresMinMax)

        if mPCA is not None:
            projMatrix = PCA(trainData, mPCA)
            trainData = np.dot(projMatrix.T, trainData)
            testData = np.dot(projMatrix.T, testData)
        
        classDict = model(
            trainData, trainLabel, numberOfClass)
        
        marginalDensities = Gaussian_classification(
            classDict, testData, numberOfClass)

        for j in range(testData.shape[1]):
            foldedLLR[i][j] = marginalDensities[1][j] - marginalDensities[0][j]

    LLR = foldedLLR.ravel()
    normalizeDCF = binaryBayesRisk(pi, Cfn, Cfp, LLR, label)[0]
    min_normalizeDCF = binaryMinDCF(pi, Cfn, Cfp, LLR, label)
    return normalizeDCF, min_normalizeDCF


def KFold_logReg_binary(D, L, K, pi, Cfn, Cfp, la, mPCA=None):
    (data, label) = shuffleData(D, L)
    idx = range(data.shape[1])
    interval = int(data.shape[1] / float(K))
    foldedLLR = np.zeros((K, interval))

    for i in range(K):
        startIndex = i * interval
        endIndex = startIndex + interval

        idxTest = idx[startIndex:endIndex]
        idxTrain = np.delete(idx, idx[startIndex:endIndex])

        trainData = data[:, idxTrain]
        testData = data[:, idxTest]
        trainLabel = label[idxTrain]

        featuresMinMax = featureNormalization(trainData)
        trainData = dataNormalization(trainData, featuresMinMax)
        testData = dataNormalization(testData, featuresMinMax)

        if mPCA is not None:
            projMatrix = PCA(trainData, mPCA)
            trainData = np.dot(projMatrix.T, trainData)
            testData = np.dot(projMatrix.T, testData)

        w, b = logisticRegression_binary_model(
            trainData, trainLabel, la)

        foldedLLR[i, :] = logisticRegression_binary_classification(
            testData, w, b)

    LLR = foldedLLR.ravel()
    normalizeDCF = binaryBayesRisk(pi, Cfn, Cfp, LLR, label)[0]
    min_normalizeDCF = binaryMinDCF(pi, Cfn, Cfp, LLR, label)
    return normalizeDCF, min_normalizeDCF


def KFold_LinearSVM(D, L, K, pi, Cfn, Cfp, C, Ksvm=1, mPCA=None):
    (data, label) = shuffleData(D, L)
    idx = range(data.shape[1])
    interval = int(data.shape[1] / float(K))
    foldedLLR = np.zeros((K, interval))

    for i in range(K):
        startIndex = i * interval
        endIndex = startIndex + interval

        idxTest = idx[startIndex:endIndex]
        idxTrain = np.delete(idx, idx[startIndex:endIndex])

        trainData = data[:, idxTrain]
        testData = data[:, idxTest]
        trainLabel = label[idxTrain]

        featuresMinMax = featureNormalization(trainData)
        trainData = dataNormalization(trainData, featuresMinMax)
        testData = dataNormalization(testData, featuresMinMax)

        if mPCA is not None:
            projMatrix = PCA(trainData, mPCA)
            trainData = np.dot(projMatrix.T, trainData)
            testData = np.dot(projMatrix.T, testData)

        wStar = LinearSVM_model(
            trainData, trainLabel, C, Ksvm)

        foldedLLR[i, :] = LinearSVM_classification(
            testData, wStar, Ksvm)

    LLR = foldedLLR.ravel()
    normalizeDCF = binaryBayesRisk(pi, Cfn, Cfp, LLR, label)[0]
    min_normalizeDCF = binaryMinDCF(pi, Cfn, Cfp, LLR, label)
    return normalizeDCF, min_normalizeDCF


def KFold_SVM_PolynomialKernel(D, L, K, pi, Cfn, Cfp, c, d, C, Ksvm=1, mPCA=None):
    (data, label) = shuffleData(D, L)
    idx = range(data.shape[1])
    interval = int(data.shape[1] / float(K))
    foldedLLR = np.zeros((K, interval))

    for i in range(K):
        startIndex = i * interval
        endIndex = startIndex + interval

        idxTest = idx[startIndex:endIndex]
        idxTrain = np.delete(idx, idx[startIndex:endIndex])

        trainData = data[:, idxTrain]
        testData = data[:, idxTest]
        trainLabel = label[idxTrain]

        featuresMinMax = featureNormalization(trainData)
        trainData = dataNormalization(trainData, featuresMinMax)
        testData = dataNormalization(testData, featuresMinMax)

        if mPCA is not None:
            projMatrix = PCA(trainData, mPCA)
            trainData = np.dot(projMatrix.T, trainData)
            testData = np.dot(projMatrix.T, testData)

        alphaStar = SVM_PolynomialKernel_model(
            trainData, trainLabel, c, d, C, Ksvm)

        foldedLLR[i, :] = SVM_PolynomialKernel_classification(
            trainData, trainLabel, testData, alphaStar, c, d, Ksvm)

    LLR = foldedLLR.ravel()
    normalizeDCF = binaryBayesRisk(pi, Cfn, Cfp, LLR, label)[0]
    min_normalizeDCF = binaryMinDCF(pi, Cfn, Cfp, LLR, label)
    return normalizeDCF, min_normalizeDCF


def KFold_SVM_RadialBasisKernel(D, L, K, pi, Cfn, Cfp, gamma, C, Ksvm=1, mPCA=None):
    (data, label) = shuffleData(D, L)
    idx = range(data.shape[1])
    interval = int(data.shape[1] / float(K))
    foldedLLR = np.zeros((K, interval))

    for i in range(K):
        startIndex = i * interval
        endIndex = startIndex + interval

        idxTest = idx[startIndex:endIndex]
        idxTrain = np.delete(idx, idx[startIndex:endIndex])

        trainData = data[:, idxTrain]
        testData = data[:, idxTest]
        trainLabel = label[idxTrain]

        featuresMinMax = featureNormalization(trainData)
        trainData = dataNormalization(trainData, featuresMinMax)
        testData = dataNormalization(testData, featuresMinMax)

        if mPCA is not None:
            projMatrix = PCA(trainData, mPCA)
            trainData = np.dot(projMatrix.T, trainData)
            testData = np.dot(projMatrix.T, testData)

        alphaStar = SVM_RadialBasisKernel_model(
            trainData, trainLabel, gamma, C, Ksvm)

        foldedLLR[i, :] = SVM_RadialBasisKernel_classification(
            trainData, trainLabel, testData, alphaStar, gamma, Ksvm)

    LLR = foldedLLR.ravel()

    # binaryDrawROC(pi, Cfn, Cfp, LLR, label)
    # drawBayesErrorPlot(LLR, label)

    normalizeDCF = binaryBayesRisk(pi, Cfn, Cfp, LLR, label)[0]
    min_normalizeDCF = binaryMinDCF(pi, Cfn, Cfp, LLR, label)
    return normalizeDCF, min_normalizeDCF


def KFold_GMM(D, L, numberOfClass, K, pi, Cfn, Cfp, numberOfComponents, alpha, delta, psi, model=None, mPCA=None):
    (data, label) = shuffleData(D, L)
    idx = range(data.shape[1])
    interval = int(data.shape[1] / float(K))
    foldedLLR = np.zeros((K, interval))

    for i in range(K):
        startIndex = i * interval
        endIndex = startIndex + interval

        idxTest = idx[startIndex:endIndex]
        idxTrain = np.delete(idx, idx[startIndex:endIndex])

        trainData = data[:, idxTrain]
        testData = data[:, idxTest]
        trainLabel = label[idxTrain]

        featuresMinMax = featureNormalization(trainData)
        trainData = dataNormalization(trainData, featuresMinMax)
        testData = dataNormalization(testData, featuresMinMax)

        if mPCA is not None:
            projMatrix = PCA(trainData, mPCA)
            trainData = np.dot(projMatrix.T, trainData)
            testData = np.dot(projMatrix.T, testData)

        classDict = GMM_model(
            trainData, trainLabel, numberOfClass, numberOfComponents, alpha, delta, psi, model)

        marginalDensities = GMM_classification(
            classDict, testData, numberOfClass)

        for j in range(testData.shape[1]):
            foldedLLR[i][j] = marginalDensities[1][j] - marginalDensities[0][j]

    LLR = foldedLLR.ravel()

    normalizeDCF = binaryBayesRisk(pi, Cfn, Cfp, LLR, label)[0]
    min_normalizeDCF = binaryMinDCF(pi, Cfn, Cfp, LLR, label)
    return normalizeDCF, min_normalizeDCF


def optimalDCF(D, L, K, pi, Cfn, Cfp, gamma, C, Ksvm=1, mPCA=None):
    (data, label) = shuffleData(D, L)
    idx = range(data.shape[1])
    interval = int(data.shape[1] / float(K))
    foldedLLR = np.zeros((K, interval))

    for i in range(K):
        startIndex = i * interval
        endIndex = startIndex + interval
        idxTest = idx[startIndex:endIndex]
        idxTrain = np.delete(idx, idx[startIndex:endIndex])
        trainData = data[:, idxTrain]
        testData = data[:, idxTest]
        trainLabel = label[idxTrain]
        featuresMinMax = featureNormalization(trainData)
        trainData = dataNormalization(trainData, featuresMinMax)
        testData = dataNormalization(testData, featuresMinMax)
        if mPCA is not None:
            projMatrix = PCA(trainData, mPCA)
            trainData = np.dot(projMatrix.T, trainData)
            testData = np.dot(projMatrix.T, testData)
        alphaStar = SVM_RadialBasisKernel_model(
            trainData, trainLabel, gamma, C, Ksvm)
        foldedLLR[i, :] = SVM_RadialBasisKernel_classification(
            trainData, trainLabel, testData, alphaStar, gamma, Ksvm)
    LLR = foldedLLR.ravel()

    np.random.seed(0)
    idx = np.random.permutation(LLR.shape[0])
    index = idx[:]
    shuffledLLR = LLR[index]
    shuffledlabel = label[index]

    idx = range(shuffledLLR.shape[0])
    trainLLR = shuffledLLR[:1200]
    trainLabels = shuffledlabel[:1200]
    testLLR = shuffledLLR[1200:]
    testLabels = shuffledlabel[1200:]

    minDCF = None
    tStar = 0
    for i in trainLLR:
        normalizeDCF = binaryBayesRisk(pi, Cfn, Cfp, trainLLR, trainLabels, i)[0]
        if minDCF == None or normalizeDCF < minDCF:
            minDCF = normalizeDCF
            tStar = i

    DCF_t = binaryBayesRisk(pi, Cfn, Cfp, testLLR, testLabels)[0]
    DCF_tStar = binaryBayesRisk(pi, Cfn, Cfp, testLLR, testLabels, tStar)[0]
    minDCF = binaryMinDCF(pi, Cfn, Cfp, testLLR, testLabels)
    return DCF_t, DCF_tStar, minDCF, tStar

