import numpy as np

def ReadData(fileName):
    f = open(fileName, 'r')
    lines = f.read().splitlines()
    f.close()

    items = []

    for i in range(len(lines)):
        line = lines[i].split(',')
        itemFeatures = []

        for j in range(len(line) - 1):
            v = float(line[j])
            itemFeatures.append(v)

        items.append(itemFeatures)

    return np.array(items)


def FindColMinMax(items):
    max = np.max(items, axis=0)
    min = np.min(items, axis=0)

    return min, max


def InitializeMeans(items, k, mins, maxs):
    means = np.random.random_sample((k, len(items[0])))
    f = len(items[0])
    means = np.multiply(means,maxs-mins)+mins
    return means


def UpdateMean(n, mean, item):
    mean = (mean*(n-1)+item)/n
    return mean


def Classify(means, item):
    dis = np.linalg.norm(means-item, axis=1)
    return np.argmin(dis)


def CalculateMeans(k, items, maxIterations=100000):
    cMin, cMax = FindColMinMax(items)
    means = InitializeMeans(items,k,cMin,cMax)
    clusterSize = [0 for x in range(k)]
    belongsTo = [0 for x in range(len(items))]

    for e in range(maxIterations):
        noChange = True
        # clusterSize = [0 for x in range(k)]         #comment this
        for i in range(len(items)):
            item = items[i]
            index = Classify(means, item)
            clusterSize[index] += 1
            means[index] = UpdateMean(clusterSize[index],means[index],item)

            if(index!=belongsTo[i]):
                noChange = False
            belongsTo[i] = index

        if (noChange):
            break

    return means


def FindClusters(means, items):
    clusters = [[] for i in range(len(means))]

    for item in items:
        clusters[Classify(means,item)].append(item)

    return clusters


def Cluster(data, k):
    means = CalculateMeans(k, data)
    clusters = FindClusters(means, data)
    return clusters
