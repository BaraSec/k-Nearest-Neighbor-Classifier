#from random import shuffle
import numpy as np

def main():
    print()
    trainingData = open(input(r"Training datafile path\name ==> "), "r")
    testData = open(input(r"Test datafile path\name ==> "), "r")

    training_data_arr = []
    test_data_arr = []
 
    for vector in trainingData.readlines():
        training_data_arr.append(vector.rstrip('\n').split(","))
 
    for vector in testData.readlines():
        test_data_arr.append(vector.rstrip('\n').split(","))
 
    #shuffle(data_arr)
 
    for i in range(0, len(training_data_arr[0])-1):
        mmNormalize(training_data_arr, max(float(maxX(training_data_arr, i)), float(maxX(test_data_arr, i))), min(float(minX(training_data_arr, i)), float(minX(test_data_arr, i))), i)
 
    for i in range(0, len(test_data_arr[0])-1):
        mmNormalize(test_data_arr, max(float(maxX(training_data_arr, i)), float(maxX(test_data_arr, i))), min(float(minX(training_data_arr, i)), float(minX(test_data_arr, i))), i)
 
    trainingDataNormalized = open("training.data_normalized", "w+")
    for i in range(0,len(training_data_arr)):
        for j in range(0, len(training_data_arr[i])):
            trainingDataNormalized.write(str(training_data_arr[i][j]))
            if j != len(training_data_arr[i])-1:
                trainingDataNormalized.write(",")
        trainingDataNormalized.write("\n")
    trainingDataNormalized.close()
 
    testDataNormalized = open("test.data_normalized", "w+")
    for i in range(0,len(test_data_arr)):
        for j in range(0, len(test_data_arr[i])):
            testDataNormalized.write(str(test_data_arr[i][j]))
            if j != len(test_data_arr[i])-1:
                testDataNormalized.write(",")
        testDataNormalized.write("\n")
    testDataNormalized.close()
 
    k = input("\nEnter K ===> ")
 
    opt = ""
 
    print("\nChoose a distance measure <1-6>:")
    print("1 - Euclidean Distance\n2 - Manhattan Distance\n3 - Chebyshev Distance\n4 - Cosine Distance\n5 - Chi-square Distance\n6 - Gower Distance")
 
    while (opt == "" or int(opt) > 6 or int(opt) < 1):
        opt = input("<1-6> ==> ")

    results = []

    if int(opt) == 1:
        results = euclideanD(training_data_arr, test_data_arr, k)
    elif int(opt) == 2:
        results = manhattanD(training_data_arr, test_data_arr, k)
    elif int(opt) == 3:
        results = chebyshevD(training_data_arr, test_data_arr, k)
    elif int(opt) == 4:
        results = cosineD(training_data_arr, test_data_arr, k)
    elif int(opt) == 5:
        results = chiSquareD(training_data_arr, test_data_arr, k)
    elif int(opt) == 6:
        results = gowerD(training_data_arr, test_data_arr, k)

    correct = 0
    fp_seto = 0
    fp_vers = 0
    fp_virg = 0
    fn_seto = 0
    fn_vers = 0
    fn_virg = 0
    tp_seto = 0
    tp_vers = 0
    tp_virg = 0
    prec_seto = 0
    prec_vers = 0
    prec_virg = 0
    recall_seto = 0
    recall_vers = 0
    recall_virg = 0

    outRes = open("testData.result", "w+")
    for i in range(0,len(test_data_arr)):
        for j in range(0, len(test_data_arr[i][:-1])):
            outRes.write(str(test_data_arr[i][j]) + ",")
        outRes.write(str(results[i]))
        outRes.write("\n")

        if str(test_data_arr[i][4]) == str(results[i]):
            correct += 1
            if str(results[i]) == "Iris-versicolor":
                tp_vers += 1
            elif str(results[i]) == "Iris-virginica":
                tp_virg += 1
            else:
                tp_seto += 1
        elif str(test_data_arr[i][4]) == "Iris-setosa" and str(results[i]) == "Iris-versicolor":
            fp_vers += 1
            fn_seto += 1
        elif str(test_data_arr[i][4]) == "Iris-setosa" and str(results[i]) == "Iris-virginica":
            fp_virg += 1
            fn_seto += 1
        elif str(test_data_arr[i][4]) == "Iris-versicolor" and str(results[i]) == "Iris-virginica":
            fp_virg += 1
            fn_vers += 1
        elif str(test_data_arr[i][4]) == "Iris-versicolor" and str(results[i]) == "Iris-setosa":
            fp_seto += 1
            fn_vers += 1
        elif str(test_data_arr[i][4]) == "Iris-virginica" and str(results[i]) == "Iris-setosa":
            fp_seto += 1
            fn_virg += 1
        elif str(test_data_arr[i][4]) == "Iris-virginica" and str(results[i]) == "Iris-versicolor":
            fp_vers += 1
            fn_virg += 1
    outRes.close()
    
    if tp_seto != 0 or fp_seto != 0:
        prec_seto = tp_seto / (tp_seto + fp_seto)
    if tp_vers != 0 or fp_vers != 0:
        prec_vers = tp_vers / (tp_vers + fp_vers)
    if tp_virg != 0 or fp_virg != 0:
        prec_virg = tp_virg / (tp_virg + fp_virg)
    if tp_seto != 0 or fn_seto != 0:
        recall_seto = tp_seto / (tp_seto + fn_seto)
    if tp_vers != 0 or fn_vers != 0:
        recall_vers = tp_vers / (tp_vers + fn_vers)
    if tp_virg != 0 or fn_virg != 0:
        recall_virg = tp_virg / (tp_virg + fn_virg)

    print("\n\n\n>>> Results were saved to file \"testData.result\"\n>>> " + str(correct)  + " out of " + str(len(results))  + " classifications are correct\n>>> Accuracy = %" + str((float(correct)) / float(len(results)) * 100.0) + "\n")

    print("\n----------------------------------------------------\n")

    print("\n\t\tPredicted - Seto\n\t\t    Negative\tPositive\nActual\tNegative\t" + str(len(test_data_arr) - (fp_seto + fn_seto + tp_seto)) + "\t" + str(fp_seto) + "\n\tPositive\t" + str(fn_seto) + "\t" + str(tp_seto))

    print("\n\n\t\tPredicted - Vers\n\t\t    Negative\tPositive\nActual\tNegative\t" + str(len(test_data_arr) - (fp_vers + fn_vers + tp_vers)) + "\t" + str(fp_vers) + "\n\tPositive\t" + str(fn_vers) + "\t" + str(tp_vers))

    print("\n\n\t\tPredicted - Virg\n\t\t    Negative\tPositive\nActual\tNegative\t" + str(len(test_data_arr) - (fp_virg + fn_virg + tp_virg)) + "\t" + str(fp_virg) + "\n\tPositive\t" + str(fn_virg) + "\t" + str(tp_virg))

    print("\n----------------------------------------------------\n")

    print("\n>>> Precision(seto) = " + str(prec_seto) + "\n>>> Precision(vers) = " + str(prec_vers) + "\n>>> Precision(virg) = " + str(prec_virg))

    print(">>> Recall(seto) = " + str(recall_seto) + "\n>>> Recall(vers) = " + str(recall_vers) + "\n>>> Recall(virg) = " + str(recall_virg) + "\n")

    print("\n>>> Precision = " + str((prec_seto + prec_vers + prec_virg) / 3))
    
    print(">>> Recall = " + str((recall_seto + recall_vers + recall_virg) / 3) + "\n")


def maxX(data_arr, i):
    max_num = data_arr[0][i]
 
    for j in range(1, len(data_arr)):
        if data_arr[j][i] > max_num:
            max_num = data_arr[j][i]
 
    return max_num
 
 
def minX(data_arr, i):
    min_num = data_arr[0][i]
 
    for j in range(1, len(data_arr)):
        if data_arr[j][i] < min_num:
            min_num = data_arr[j][i]
 
    return min_num
 
 
def mmNormalize(data_arr, max, min, i):
    for j in range(0, len(data_arr)):
        data_arr[j][i] = (float(data_arr[j][i]) - float(min)) / (float(max) - float(min))
 
def euclideanD(trainingD, testD, k):
    fResults = []
    
    for i in range(0, len(testD)):
        iResults = []
        l1 = 0
        l2 = 0
        l3 = 0
        for j in range(0, len(trainingD)):
            for l in range(0, len(testD[i])-1):
                if l != 0:
                    iResults.insert(j, [float(iResults.pop()[0]) + ((testD[i][l] - trainingD[j][l]) ** 2)])
                else:
                    iResults.insert(j, [(testD[i][l] - trainingD[j][l]) ** 2])
            iResults[j][0] = iResults[j][0] ** 0.5
            iResults[j].append(trainingD[j][len(trainingD[j])-1])
        iResults.sort()

        for n in range(0, int(k)):
            if iResults[n][1] == "Iris-setosa":
                l1 += 1
            elif iResults[n][1] == "Iris-versicolor":
                l2 += 1
            else:
                l3 += 1
 
        if l1 > l2 and l1 > l3:
            fResults.append("Iris-setosa")
        elif l2 > l1 and l2 > l3:
            fResults.append("Iris-versicolor")
        else:
            fResults.append("Iris-virginica")
        
    return fResults


def manhattanD(trainingD, testD, k):
    fResults = []

    for i in range(0, len(testD)):
        iResults = []
        l1 = 0
        l2 = 0
        l3 = 0
        for j in range(0, len(trainingD)):
            for l in range(0, len(testD[i])-1):
                if l != 0:
                    iResults.insert(j, [float(iResults.pop()[0]) + abs(testD[i][l] - trainingD[j][l])])
                else:
                    iResults.insert(j, [abs(testD[i][l] - trainingD[j][l])])
            iResults[j].append(trainingD[j][len(trainingD[j])-1])
        iResults.sort()

        for n in range(0, int(k)):
            if iResults[n][1] == "Iris-setosa":
                l1 += 1
            elif iResults[n][1] == "Iris-versicolor":
                l2 += 1
            else:
                l3 += 1

        if l1 > l2 and l1 > l3:
            fResults.append("Iris-setosa")
        elif l2 > l1 and l2 > l3:
            fResults.append("Iris-versicolor")
        else:
            fResults.append("Iris-virginica")

    return fResults


def chebyshevD(trainingD, testD, k):
    fResults = []

    for i in range(0, len(testD)):
        iResults = []
        l1 = 0
        l2 = 0
        l3 = 0
        for j in range(0, len(trainingD)):
            for l in range(0, len(testD[i])-1):
                if l != 0:
                    if float(iResults[j][0]) < abs(float(testD[i][l]) - float(trainingD[j][l])):
                        iResults.pop()
                        iResults.insert(j, [abs(testD[i][l] - trainingD[j][l])])
                else:
                    iResults.insert(j, [abs(testD[i][l] - trainingD[j][l])])
            iResults[j].append(trainingD[j][len(trainingD[j])-1])
        iResults.sort()

        for n in range(0, int(k)):
            if iResults[n][1] == "Iris-setosa":
                l1 += 1
            elif iResults[n][1] == "Iris-versicolor":
                l2 += 1
            else:
                l3 += 1

        if l1 > l2 and l1 > l3:
            fResults.append("Iris-setosa")
        elif l2 > l1 and l2 > l3:
            fResults.append("Iris-versicolor")
        else:
            fResults.append("Iris-virginica")

    return fResults


def cosineD(trainingD, testD, k):
    fResults = []

    for i in range(0, len(testD)):
        iResults = []
        l1 = 0
        l2 = 0
        l3 = 0
        for j in range(0, len(trainingD)):
            iResults.insert(j, [np.dot(testD[i][:-1], trainingD[j][:-1]) / (np.sqrt(np.dot(testD[i][:-1], testD[i][:-1])) * np.sqrt(np.dot(trainingD[j][:-1], trainingD[j][:-1])))])
            iResults[j].append(trainingD[j][len(trainingD[j])-1])
        iResults.sort(reverse=True)

        for n in range(0, int(k)):
            if iResults[n][1] == "Iris-setosa":
                l1 += 1
            elif iResults[n][1] == "Iris-versicolor":
                l2 += 1
            else:
                l3 += 1

        if l1 > l2 and l1 > l3:
            fResults.append("Iris-setosa")
        elif l2 > l1 and l2 > l3:
            fResults.append("Iris-versicolor")
        else:
            fResults.append("Iris-virginica")

    return fResults


def chiSquareD(trainingD, testD, k):
    fResults = []

    for i in range(0, len(testD)):
        iResults = []
        l1 = 0
        l2 = 0
        l3 = 0
        for j in range(0, len(trainingD)):
            for l in range(0, len(testD[i])-1):
                if l != 0:
                    iResults.insert(j, [float(iResults.pop()[0]) + (((testD[i][l] - trainingD[j][l]) ** 2) / (testD[i][l] + trainingD[j][l]))])
                else:
                    iResults.insert(j, [((testD[i][l] - trainingD[j][l]) ** 2) / (testD[i][l] + trainingD[j][l])])
            iResults[j][0] = iResults[j][0] * 0.5
            iResults[j].append(trainingD[j][len(trainingD[j])-1])
        iResults.sort()

        for n in range(0, int(k)):
            if iResults[n][1] == "Iris-setosa":
                l1 += 1
            elif iResults[n][1] == "Iris-versicolor":
                l2 += 1
            else:
                l3 += 1

        if l1 > l2 and l1 > l3:
            fResults.append("Iris-setosa")
        elif l2 > l1 and l2 > l3:
            fResults.append("Iris-versicolor")
        else:
            fResults.append("Iris-virginica")

    return fResults


def gowerD(trainingD, testD, k):
    fResults = []

    for i in range(0, len(testD)):
        iResults = []
        l1 = 0
        l2 = 0
        l3 = 0
        for j in range(0, len(trainingD)):
            for l in range(0, len(testD[i])-1):
                if l != 0:
                    iResults.insert(j, [float(iResults.pop()[0]) + (abs(testD[i][l] - trainingD[j][l]) / (max(float(maxX(testD, l)), float(maxX(trainingD, l))) - min(float(minX(testD, l)), float(minX(trainingD, l)))))])
                else:
                    iResults.insert(j, [abs(testD[i][l] - trainingD[j][l]) / (max(float(maxX(testD, l)), float(maxX(trainingD, l))) - min(float(minX(testD, l)), float(minX(trainingD, l))))])
            iResults[j][0] = iResults[j][0] / len(trainingD[j])
            iResults[j].append(trainingD[j][len(trainingD[j])-1])
        iResults.sort()

        for n in range(0, int(k)):
            if iResults[n][1] == "Iris-setosa":
                l1 += 1
            elif iResults[n][1] == "Iris-versicolor":
                l2 += 1
            else:
                l3 += 1

        if l1 > l2 and l1 > l3:
            fResults.append("Iris-setosa")
        elif l2 > l1 and l2 > l3:
            fResults.append("Iris-versicolor")
        else:
            fResults.append("Iris-virginica")

    return fResults


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        exit()
