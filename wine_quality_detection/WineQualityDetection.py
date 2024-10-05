from myLibrary import *

labelDictionary = {
    0: 'Low quality',
    1: 'High quality'
}

featureDictionary = {
    0: 'fixed acidity',
    1: 'volatile acidity',
    2: 'citric acid',
    3: 'residual sugar',
    4: 'chlorides',
    5: 'free sulfur dioxide',
    6: 'total sulfur dioxide',
    7: 'density',
    8: 'pH',
    9: 'sulphates',
    10: 'alcohol'
}

# ********************************************* Load *********************************************

data, label = load('Data/Train.txt')

# featuresMinMax = featureNormalization(data)
# normalizedData = dataNormalization(data, featuresMinMax)

# print(normalizedData)

# wStar = LinearSVM_model(data, label, C=100, K=1)
# SVMscore = LinearSVM_classification(data, wStar, K=1)
# print(wStar)
# print(SVMscore)

# ********************************************* Plot *********************************************

# lowQuality = extractLowQuality(data, label)
# highQuality = extractHighQuality(data, label)    # => Pi = 613/1839
# for i in range(11):
#     plt.figure()
#     plt.xlabel(f'{featureDictionary[i]}')
#     plt.hist(lowQuality[i, :], bins=50, density=True, alpha=0.4, label='Low')
#     plt.hist(highQuality[i, :], bins=50, density=True, alpha=0.4, label='High')
#     plt.legend()
# plt.show()

# ********************************************* PCA *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedData = dataNormalization(data, featuresMinMax)
# projMatrix = PCA(normalizedData, 2)
# DP = np.dot(projMatrix.T, normalizedData)
# lowQuality = extractLowQuality(DP, label)
# highQuality = extractHighQuality(DP, label)
# plt.figure()
# plt.scatter(lowQuality[0, :], lowQuality[1, :], label = 'Low Quality')
# plt.scatter(highQuality[0, :], highQuality[1, :], label = 'High Quality')
# plt.legend()
# plt.show()

# ********************************************* Multivariate Gaussian *********************************************

# normalizeDCF, min_normalizeDCF = KFold_Gaussian(data, label, 2, 3, pi=0.5, Cfn=1, Cfp=1, model=multiVariateGaussian_model, mPCA=8)
# print(min_normalizeDCF)

# Without PCA: PI=0.5: minDCF=0.324, PI=0.1: minDCF=0.788, PI=0.9: minDCF=0.878                  THE BEST
# m=10:        PI=0.5: minDCF=0.341, PI=0.1: minDCF=0.813, PI=0.9: minDCF=0.862
# m=9:         PI=0.5: minDCF=0.335, PI=0.1: minDCF=0.831, PI=0.9: minDCF=0.887
# m=8:         PI=0.5: minDCF=0.320, PI=0.1: minDCF=0.810, PI=0.9: minDCF=0.804     THE BEST
# m=7:         PI=0.5: minDCF=0.334, PI=0.1: minDCF=0.839, PI=0.9: minDCF=0.791                               THE BEST
# m=6:         PI=0.5: minDCF=0.378, PI=0.1: minDCF=0.812, PI=0.9: minDCF=0.917

# ********************************************* Naive Bayes Gaussian *********************************************

# normalizeDCF, min_normalizeDCF = KFold_Gaussian(data, label, 2, 3, pi=0.5, Cfn=1, Cfp=1, model=naiveBayesGaussian_model, mPCA=7)
# print(min_normalizeDCF)

# Without PCA: PI=0.5: minDCF=0.427, PI=0.1: minDCF=0.857, PI=0.9: minDCF=0.896
# m=10:        PI=0.5: minDCF=0.405, PI=0.1: minDCF=0.831, PI=0.9: minDCF=0.962                  THE BEST
# m=9:         PI=0.5: minDCF=0.403, PI=0.1: minDCF=0.831, PI=0.9: minDCF=0.966
# m=8:         PI=0.5: minDCF=0.392, PI=0.1: minDCF=0.841, PI=0.9: minDCF=0.855                               THE BEST
# m=7:         PI=0.5: minDCF=0.386, PI=0.1: minDCF=0.876, PI=0.9: minDCF=0.918     THE BEST
# m=6:         PI=0.5: minDCF=0.406, PI=0.1: minDCF=0.840, PI=0.9: minDCF=0.949

# ********************************************* Tied Covariance Gaussian *********************************************

# normalizeDCF, min_normalizeDCF = KFold_Gaussian(data, label, 2, 3, pi=0.5, Cfn=1, Cfp=1, model=tiedCovGaussian_model, mPCA=10)
# print(min_normalizeDCF)

# Without PCA: PI=0.5: minDCF=0.337, PI=0.1: minDCF=0.817, PI=0.9: minDCF=0.743                  THE BEST     THE BEST
# m=10:        PI=0.5: minDCF=0.336, PI=0.1: minDCF=0.826, PI=0.9: minDCF=0.753     THE BEST
# m=9:         PI=0.5: minDCF=0.336, PI=0.1: minDCF=0.825, PI=0.9: minDCF=0.759
# m=8:         PI=0.5: minDCF=0.344, PI=0.1: minDCF=0.835, PI=0.9: minDCF=0.765
# m=7:         PI=0.5: minDCF=0.367, PI=0.1: minDCF=0.852, PI=0.9: minDCF=0.827
# m=6:         PI=0.5: minDCF=0.388, PI=0.1: minDCF=0.829, PI=0.9: minDCF=0.888

# ********************************************* Tied Covariance Naive Bayes Gaussian *********************************************

# normalizeDCF, min_normalizeDCF = KFold_Gaussian(data, label, 2, 3, pi=0.5, Cfn=1, Cfp=1, model=tiedNaiveBayesGaussian_model, mPCA=6)
# print(min_normalizeDCF)

# Without PCA: PI=0.5: minDCF=0.405, PI=0.1: minDCF=0.863, PI=0.9: minDCF=0.922
# m=10:        PI=0.5: minDCF=0.343, PI=0.1: minDCF=0.830, PI=0.9: minDCF=0.775                               THE BEST
# m=9:         PI=0.5: minDCF=0.342, PI=0.1: minDCF=0.813, PI=0.9: minDCF=0.788     THE BEST     THE BEST
# m=8:         PI=0.5: minDCF=0.345, PI=0.1: minDCF=0.833, PI=0.9: minDCF=0.782
# m=7:         PI=0.5: minDCF=0.359, PI=0.1: minDCF=0.854, PI=0.9: minDCF=0.822
# m=6:         PI=0.5: minDCF=0.386, PI=0.1: minDCF=0.829, PI=0.9: minDCF=0.891

# ********************************************* Logistic Regression *********************************************

# normalizeDCF, min_normalizeDCF = KFold_logReg_binary(data, label, 3, pi=0.5, Cfn=1, Cfp=1, la=1e-6, mPCA=None)
# print(min_normalizeDCF)

# Without PCA, la=1e-6: PI=0.5: minDCF=0.346, PI=0.1: minDCF=0.846, PI=0.9: minDCF=0.695     THE BEST
# Without PCA, la=1e-3: PI=0.5: minDCF=0.349, PI=0.1: minDCF=0.827, PI=0.9: minDCF=0.789                  THE BEST    
# Without PCA, la=1e-1: PI=0.5: minDCF=0.422, PI=0.1: minDCF=0.875, PI=0.9: minDCF=0.980
# Without PCA, la=1:    PI=0.5: minDCF=0.637, PI=0.1: minDCF=0.948, PI=0.9: minDCF=0.982
# m=10, la=1e-6:        PI=0.5: minDCF=0.347, PI=0.1: minDCF=0.843, PI=0.9: minDCF=0.690
# m=9, la=1e-6:         PI=0.5: minDCF=0.349, PI=0.1: minDCF=0.851, PI=0.9: minDCF=0.693
# m=8, la=1e-6:         PI=0.5: minDCF=0.347, PI=0.1: minDCF=0.851, PI=0.9: minDCF=0.689                               THE BEST
# m=7, la=1e-6:         PI=0.5: minDCF=0.377, PI=0.1: minDCF=0.867, PI=0.9: minDCF=0.777

# ********************************************* Linear SVM *********************************************

# normalizeDCF, min_normalizeDCF = KFold_LinearSVM(data, label, 3, pi=0.5, Cfn=1, Cfp=1, C=100, Ksvm=1, mPCA=None)
# print(min_normalizeDCF)

# Without PCA, C=1:   PI=0.5: minDCF=0.338, PI=0.1: minDCF=0.816, PI=0.9: minDCF=0.805                  THE BEST
# Without PCA, C=0.1: PI=0.5: minDCF=0.365, PI=0.1: minDCF=0.835, PI=0.9: minDCF=0.869
# Without PCA, C=10:  PI=0.5: minDCF=0.338, PI=0.1: minDCF=0.822, PI=0.9: minDCF=0.760
# Without PCA, C=100: PI=0.5: minDCF=0.334, PI=0.1: minDCF=0.823, PI=0.9: minDCF=0.723     THE BEST                  THE BEST
# m=10, C=1:          PI=0.5: minDCF=0.341, PI=0.1: minDCF=0.821, PI=0.9: minDCF=0.789
# m=9, C=1:           PI=0.5: minDCF=0.340, PI=0.1: minDCF=0.822, PI=0.9: minDCF=0.790
# m=8, C=1:           PI=0.5: minDCF=0.347, PI=0.1: minDCF=0.822, PI=0.9: minDCF=0.794
# m=7, C=1:           PI=0.5: minDCF=0.367, PI=0.1: minDCF=0.836, PI=0.9: minDCF=0.811

# ********************************************* Polynomial SVM *********************************************

# normalizeDCF, min_normalizeDCF = KFold_SVM_PolynomialKernel(data, label, 3, pi=0.5, Cfn=1, Cfp=1, c=1, d=3, C=100, Ksvm=1, mPCA=None)
# print(min_normalizeDCF)

# Without PCA, c=0, d=2, C=1:   PI=0.5: minDCF=0.338, PI=0.1: minDCF=0.809, PI=0.9: minDCF=0.759
# Without PCA, c=0, d=2, C=0.1: PI=0.5: minDCF=0.359, PI=0.1: minDCF=0.824, PI=0.9: minDCF=0.874
# Without PCA, c=0, d=2, C=10:  PI=0.5: minDCF=0.306, PI=0.1: minDCF=0.772, PI=0.9: minDCF=0.776
# Without PCA, c=0, d=2, C=100: PI=0.5: minDCF=0.286, PI=0.1: minDCF=0.746, PI=0.9: minDCF=0.803
# Without PCA, c=1, d=2, C=100: PI=0.5: minDCF=0.281, PI=0.1: minDCF=0.743, PI=0.9: minDCF=0.785
# Without PCA, c=1, d=3, C=100: PI=0.5: minDCF=0.258, PI=0.1: minDCF=0.639, PI=0.9: minDCF=0.753     THE BEST     THE BEST     THE BEST
# Without PCA, c=2, d=2, C=100: PI=0.5: minDCF=0.285, PI=0.1: minDCF=0.745, PI=0.9: minDCF=0.780
# m=10, c=0, d=2, C=1:          PI=0.5: minDCF=0.342, PI=0.1: minDCF=0.813, PI=0.9: minDCF=0.753
# m=9, c=0, d=2, C=1:           PI=0.5: minDCF=0.343, PI=0.1: minDCF=0.816, PI=0.9: minDCF=0.773
# m=8, c=0, d=2, C=1:           PI=0.5: minDCF=0.344, PI=0.1: minDCF=0.809, PI=0.9: minDCF=0.772
# m=7, c=0, d=2, C=1:           PI=0.5: minDCF=0.360, PI=0.1: minDCF=0.828, PI=0.9: minDCF=0.812

# ********************************************* Radial Basis SVM *********************************************

# normalizeDCF, min_normalizeDCF = KFold_SVM_RadialBasisKernel(data, label, 3, pi=0.5, Cfn=1, Cfp=1, gamma=1, C=100, Ksvm=1, mPCA=None)
# print(min_normalizeDCF)

# Without PCA, gamma=1, C=1:     PI=0.5: minDCF=0.317, PI=0.1: minDCF=0.775, PI=0.9: minDCF=0.671
# Without PCA, gamma=1, C=0.1:   PI=0.5: minDCF=0.365, PI=0.1: minDCF=0.815, PI=0.9: minDCF=0.758
# Without PCA, gamma=1, C=10:    PI=0.5: minDCF=0.261, PI=0.1: minDCF=0.690, PI=0.9: minDCF=0.718
# Without PCA, gamma=1, C=100:   PI=0.5: minDCF=0.250, PI=0.1: minDCF=0.611, PI=0.9: minDCF=0.723     THE BEST     THE BEST
# Without PCA, gamma=0.1, C=100: PI=0.5: minDCF=0.311, PI=0.1: minDCF=0.773, PI=0.9: minDCF=0.705
# Without PCA, gamma=10, C=100:  PI=0.5: minDCF=0.280, PI=0.1: minDCF=0.688, PI=0.9: minDCF=0.807
# m=10, gamma=1, C=1:            PI=0.5: minDCF=0.318, PI=0.1: minDCF=0.773, PI=0.9: minDCF=0.662                               THE BEST
# m=9, gamma=1, C=1:             PI=0.5: minDCF=0.320, PI=0.1: minDCF=0.789, PI=0.9: minDCF=0.673
# m=8, gamma=1, C=1:             PI=0.5: minDCF=0.320, PI=0.1: minDCF=0.793, PI=0.9: minDCF=0.672
# m=7, gamma=1, C=1:             PI=0.5: minDCF=0.345, PI=0.1: minDCF=0.776, PI=0.9: minDCF=0.719

# ********************************************* GMM *********************************************

# normalizeDCF, min_normalizeDCF = KFold_GMM(data, label, 2, 3, pi=0.5, Cfn=1, Cfp=1, numberOfComponents=16, alpha=0.1, delta=1e-6, psi=0.01, mPCA=None)
# print(min_normalizeDCF)

# Without PCA, GMMs=2:  PI=0.5: minDCF=0.319, PI=0.1: minDCF=0.805, PI=0.9: minDCF=0.838
# Without PCA, GMMs=4:  PI=0.5: minDCF=0.334, PI=0.1: minDCF=0.821, PI=0.9: minDCF=0.691
# Without PCA, GMMs=8:  PI=0.5: minDCF=0.298, PI=0.1: minDCF=0.774, PI=0.9: minDCF=0.631                  THE BEST     THE BEST
# Without PCA, GMMs=16: PI=0.5: minDCF=0.290, PI=0.1: minDCF=0.785, PI=0.9: minDCF=0.638     THE BEST
# m=10, GMMs=2:         PI=0.5: minDCF=0.320, PI=0.1: minDCF=0.799, PI=0.9: minDCF=0.837
# m=9, GMMs=2:          PI=0.5: minDCF=0.324, PI=0.1: minDCF=0.800, PI=0.9: minDCF=0.834
# m=8, GMMs=2:          PI=0.5: minDCF=0.328, PI=0.1: minDCF=0.803, PI=0.9: minDCF=0.837
# m=7, GMMs=2:          PI=0.5: minDCF=0.345, PI=0.1: minDCF=0.839, PI=0.9: minDCF=0.882

# ********************************************* GMM(tied) *********************************************

# normalizeDCF, min_normalizeDCF = KFold_GMM(data, label, 2, 3, pi=0.5, Cfn=1, Cfp=1, numberOfComponents=8, alpha=0.1, delta=1e-6, psi=0.01, model='tied', mPCA=None)
# print(min_normalizeDCF)

# Without PCA, GMMs=2:  PI=0.5: minDCF=0.339, PI=0.1: minDCF=0.798, PI=0.9: minDCF=0.844
# Without PCA, GMMs=4:  PI=0.5: minDCF=0.329, PI=0.1: minDCF=0.780, PI=0.9: minDCF=0.785                  THE BEST
# Without PCA, GMMs=8:  PI=0.5: minDCF=0.307, PI=0.1: minDCF=0.826, PI=0.9: minDCF=0.786     THE BEST
# Without PCA, GMMs=16: PI=0.5: minDCF=0.315, PI=0.1: minDCF=0.816, PI=0.9: minDCF=0.653                               THE BEST
# m=10, GMMs=2:         PI=0.5: minDCF=0.343, PI=0.1: minDCF=0.799, PI=0.9: minDCF=0.845
# m=9, GMMs=2:          PI=0.5: minDCF=0.348, PI=0.1: minDCF=0.799, PI=0.9: minDCF=0.843
# m=8, GMMs=2:          PI=0.5: minDCF=0.349, PI=0.1: minDCF=0.801, PI=0.9: minDCF=0.843
# m=7, GMMs=2:          PI=0.5: minDCF=0.347, PI=0.1: minDCF=0.827, PI=0.9: minDCF=0.864

# ********************************************* GMM(diagonal) *********************************************

# normalizeDCF, min_normalizeDCF = KFold_GMM(data, label, 2, 3, pi=0.5, Cfn=1, Cfp=1, numberOfComponents=8, alpha=0.1, delta=1e-6, psi=0.01, model='diagonal', mPCA=10)
# print(min_normalizeDCF)

# Without PCA, GMMs=2:  PI=0.5: minDCF=0.405, PI=0.1: minDCF=0.863, PI=0.9: minDCF=0.908
# m=10, GMMs=2:         PI=0.5: minDCF=0.370, PI=0.1: minDCF=0.837, PI=0.9: minDCF=0.861
# m=10, GMMs=4:         PI=0.5: minDCF=0.343, PI=0.1: minDCF=0.790, PI=0.9: minDCF=0.758
# m=10, GMMs=8:         PI=0.5: minDCF=0.319, PI=0.1: minDCF=0.803, PI=0.9: minDCF=0.642     THE BEST                  THE BEST
# m=10, GMMs=16:        PI=0.5: minDCF=0.325, PI=0.1: minDCF=0.785, PI=0.9: minDCF=0.701                  THE BEST
# m=9, GMMs=2:          PI=0.5: minDCF=0.371, PI=0.1: minDCF=0.839, PI=0.9: minDCF=0.861
# m=8, GMMs=2:          PI=0.5: minDCF=0.372, PI=0.1: minDCF=0.843, PI=0.9: minDCF=0.860
# m=7, GMMs=2:          PI=0.5: minDCF=0.380, PI=0.1: minDCF=0.855, PI=0.9: minDCF=0.910

# ********************************************* Final Evaluations on Model *********************************************

# normalizeDCF, min_normalizeDCF = KFold_SVM_RadialBasisKernel(data, label, 3, pi=0.5, Cfn=1, Cfp=1, gamma=1, C=100, Ksvm=1, mPCA=None)
# print(normalizeDCF)
# print(min_normalizeDCF)

# pi=0.5: DCF=0.263, minDCF=0.250
# pi=0.1: DCF=0.833, minDCF=0.611
# pi=0.9: DCF=0.783, minDCF=0.723

# without normalization
# pi=0.5: minDCF=0.582
# pi=0.1: minDCF=0.692
# pi=0.9: minDCF=0.740

DCF_t, DCF_tStar, minDCF, tStar = optimalDCF(data, label, 3, pi=0.5, Cfn=1, Cfp=1, gamma=1, C=100, Ksvm=1, mPCA=None)
# print(DCF_t)
# print(tStar)
# print(minDCF)

# pi=0.5: minDCF=0.243, DCF_t=0.252, DCF_tStar=0.272, tStar=-0.514
# pi=0.1: minDCF=0.485, DCF_t=0.792, DCF_tStar=0.507, tStar=+0.803
# pi=0.9: minDCF=0.648, DCF_t=0.727, DCF_tStar=0.795, tStar=-1.398

# ********************************************* Build model and results on test set *********************************************

testData, testLabel = load('Data/Test.txt')

featuresMinMax = featureNormalization(data)
normalizedTrainData = dataNormalization(data, featuresMinMax)
normalizedTestData = dataNormalization(testData, featuresMinMax)

alphaStar = SVM_RadialBasisKernel_model(normalizedTrainData, label, gamma=1, C=100, K=1)
score = SVM_RadialBasisKernel_classification(normalizedTrainData, label, normalizedTestData, alphaStar, gamma=1, K=1)
application = (0.5, 1, 1)
predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score, tStar)
errorRate = computeError(predictedLabel, testLabel)
normalizeDCF = binaryBayesRisk(application[0], application[1], application[2], score, testLabel, tStar)[0]
min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

print(f'Application: {application}')
print(f'    Error Rate: {errorRate}')
print(f'    DCF: {normalizeDCF}')
print(f'    min DCF: {min_normalizeDCF}')
# drawBayesErrorPlot(score, testLabel)

# Error Rate: 12.843029637760706
# Application: (0.5, 1, 1)
#     DCF: 0.2888054809913228
#     min DCF: 0.26887602222360946
# Application: (0.1, 1, 1)
#     DCF: 0.7843420313377865
#     min DCF: 0.6361040014982208
# Application: (0.9, 1, 1)
#     DCF: 0.6414362111659072
#     min DCF: 0.5155830576190774

# ********************************************* Multivariate Gaussian *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# projMatrix = PCA(normalizedTrainData, 8)
# reducedTrainData = np.dot(projMatrix.T, normalizedTrainData)
# reducedTestData = np.dot(projMatrix.T, normalizedTestData)

# classDict = multiVariateGaussian_model(reducedTrainData, label, 2)
# marginalDensities = Gaussian_classification(classDict, reducedTestData, 2)
# score = np.zeros(marginalDensities.shape[1])
# for i in range(marginalDensities.shape[1]):
#     score[i] = marginalDensities[1][i] - marginalDensities[0][i]
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 18.33150384193194
# Application: (0.5, 1, 1)
#     min DCF: 0.3255248975175312
# Application: (0.1, 1, 1)
#     min DCF: 0.6935748174043324
# Application: (0.9, 1, 1)
#     min DCF: 0.6971565016542857

# ********************************************* Naive Bayes Gaussian *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# projMatrix = PCA(normalizedTrainData, 7)
# reducedTrainData = np.dot(projMatrix.T, normalizedTrainData)
# reducedTestData = np.dot(projMatrix.T, normalizedTestData)

# classDict = naiveBayesGaussian_model(reducedTrainData, label, 2)
# marginalDensities = Gaussian_classification(classDict, reducedTestData, 2)
# score = np.zeros(marginalDensities.shape[1])
# for i in range(marginalDensities.shape[1]):
#     score[i] = marginalDensities[1][i] - marginalDensities[0][i]
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 17.343578485181123
# Application: (0.5, 1, 1)
#     min DCF: 0.33694362944003997
# Application: (0.1, 1, 1)
#     min DCF: 0.7568824520881452
# Application: (0.9, 1, 1)
#     min DCF: 0.695915787502341

# ********************************************* Tied Covariance Gaussian *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# projMatrix = PCA(normalizedTrainData, 10)
# reducedTrainData = np.dot(projMatrix.T, normalizedTrainData)
# reducedTestData = np.dot(projMatrix.T, normalizedTestData)

# classDict = tiedCovGaussian_model(reducedTrainData, label, 2)
# marginalDensities = Gaussian_classification(classDict, reducedTestData, 2)
# score = np.zeros(marginalDensities.shape[1])
# for i in range(marginalDensities.shape[1]):
#     score[i] = marginalDensities[1][i] - marginalDensities[0][i]
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 16.41053787047201
# Application: (0.5, 1, 1)
#     min DCF: 0.32652631250390163
# Application: (0.1, 1, 1)
#     min DCF: 0.7106795055871152
# Application: (0.9, 1, 1)
#     min DCF: 0.7288064173793619

# ********************************************* Tied Covariance Naive Bayes Gaussian *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# projMatrix = PCA(normalizedTrainData, 9)
# reducedTrainData = np.dot(projMatrix.T, normalizedTrainData)
# reducedTestData = np.dot(projMatrix.T, normalizedTestData)

# classDict = tiedNaiveBayesGaussian_model(reducedTrainData, label, 2)
# marginalDensities = Gaussian_classification(classDict, reducedTestData, 2)
# score = np.zeros(marginalDensities.shape[1])
# for i in range(marginalDensities.shape[1]):
#     score[i] = marginalDensities[1][i] - marginalDensities[0][i]
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 16.739846322722286
# Application: (0.5, 1, 1)
#     min DCF: 0.32843290259483526
# Application: (0.1, 1, 1)
#     min DCF: 0.7289390723515824
# Application: (0.9, 1, 1)
#     min DCF: 0.7441995963127119

# ********************************************* Logistic Regression *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# w, b = logisticRegression_binary_model(normalizedTrainData, label, 1e-6)
# score = logisticRegression_binary_classification(normalizedTestData, w, b)
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 15.367727771679473
# Application: (0.5, 1, 1)
#     min DCF: 0.32572518051480537
# Application: (0.1, 1, 1)
#     min DCF: 0.6948935638928772
# Application: (0.9, 1, 1)
#     min DCF: 0.6538641613084463

# ********************************************* Linear SVM *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# wStar = LinearSVM_model(normalizedTrainData, label, C=100, K=1)
# score = LinearSVM_classification(normalizedTestData, wStar, K=1)
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 14.983534577387491
# Application: (0.5, 1, 1)
#     min DCF: 0.31953461514451587
# Application: (0.1, 1, 1)
#     min DCF: 0.6675432299144765
# Application: (0.9, 1, 1)
#     min DCF: 0.6706853320015815

# ********************************************* Polynomial SVM *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# wStar = alphaStar = SVM_PolynomialKernel_model(normalizedTrainData, label, c=1, d=3, C=100, K=1)
# score = SVM_PolynomialKernel_classification(normalizedTrainData, label, normalizedTestData, alphaStar, c=1, d=3, K=1)
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 13.446761800219543
# Application: (0.5, 1, 1)
#     min DCF: 0.2671489065068564
# Application: (0.1, 1, 1)
#     min DCF: 0.6359167238903802
# Application: (0.9, 1, 1)
#     min DCF: 0.5137050793848972

# ********************************************* GMM *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# classDict = GMM_model(normalizedTrainData, label, numberOfClass=2, numberOfComponents=16, alpha=0.1, delta=1e-6, psi=0.01)
# marginalDensities = GMM_classification(classDict, normalizedTestData, numberOfClass=2)
# score = np.zeros(marginalDensities.shape[1])
# for i in range(marginalDensities.shape[1]):
#     score[i] = marginalDensities[1][i] - marginalDensities[0][i]
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 16.35565312843029
# Application: (0.5, 1, 1)
#     min DCF: 0.31516220321701316
# Application: (0.1, 1, 1)
#     min DCF: 0.6927164617017292
# Application: (0.9, 1, 1)
#     min DCF: 0.6063684790977382

# ********************************************* GMM(tied) *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# classDict = GMM_model(normalizedTrainData, label, numberOfClass=2, numberOfComponents=8, alpha=0.1, delta=1e-6, psi=0.01, model='tied')
# marginalDensities = GMM_classification(classDict, normalizedTestData, numberOfClass=2)
# score = np.zeros(marginalDensities.shape[1])
# for i in range(marginalDensities.shape[1]):
#     score[i] = marginalDensities[1][i] - marginalDensities[0][i]
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 15.14818880351262
# Application: (0.5, 1, 1)
#     min DCF: 0.28508333853548906
# Application: (0.1, 1, 1)
#     min DCF: 0.7699762781696735
# Application: (0.9, 1, 1)
#     min DCF: 0.6627260336683521

# ********************************************* GMM(diagonal) *********************************************

# featuresMinMax = featureNormalization(data)
# normalizedTrainData = dataNormalization(data, featuresMinMax)
# normalizedTestData = dataNormalization(testData, featuresMinMax)

# projMatrix = PCA(normalizedTrainData, 10)
# reducedTrainData = np.dot(projMatrix.T, normalizedTrainData)
# reducedTestData = np.dot(projMatrix.T, normalizedTestData)

# classDict = GMM_model(reducedTrainData, label, numberOfClass=2, numberOfComponents=8, alpha=0.1, delta=1e-6, psi=0.01, model='diagonal')
# marginalDensities = GMM_classification(classDict, reducedTestData, numberOfClass=2)
# score = np.zeros(marginalDensities.shape[1])
# for i in range(marginalDensities.shape[1]):
#     score[i] = marginalDensities[1][i] - marginalDensities[0][i]
# application = (0.5, 1, 1)
# predictedLabel = binaryOptimalBayesDecision(application[0], application[1], application[2], score)
# errorRate = computeError(predictedLabel, testLabel)
# min_normalizeDCF = binaryMinDCF(application[0], application[1], application[2], score, testLabel)

# print(f'Application: {application}')
# print(f'    Error Rate: {errorRate}')
# print(f'    min DCF: {min_normalizeDCF}')

# Error Rate: 16.300768386388587
# Application: (0.5, 1, 1)
#     min DCF: 0.3155705724452213
# Application: (0.1, 1, 1)
#     min DCF: 0.771997315687621
# Application: (0.9, 1, 1)
#     min DCF: 0.717655596479181