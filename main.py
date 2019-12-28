import numpy as np
from tqdm import tqdm

from nupic.encoders.scalar import ScalarEncoder
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory
from nupic.algorithms.sdr_classifier import SDRClassifier

N = 900
x = np.sin(np.arange(N)*2*np.pi/30.0)
inputDimensions = (256,)
columnDimensions = (512,)

encoder = ScalarEncoder(21, -1.0, 1.0, n=inputDimensions[0])
sp = SpatialPooler(inputDimensions=inputDimensions, columnDimensions=columnDimensions, globalInhibition=True, numActiveColumnsPerInhArea=21)
tm = TemporalMemory(columnDimensions=columnDimensions)
c = SDRClassifier(steps=[1], alpha=0.1, actValueAlpha=0.1, verbosity=0)

x_true = x[1:]
x_predict = np.zeros(len(x)-1)

for i, xi in tqdm(enumerate(x[:-1])):
    encoded = encoder.encode(xi)
    bucketIdx = np.where(encoded> 0)[0][0]
    spd = np.zeros(columnDimensions[0])
    sp.compute(encoded, True, spd)
    active_indices = np.where(spd > 0)[0]
    tm.compute(active_indices)

    active_cell_indices = tm.getActiveCells()
    predictive_cell_indices = tm.getPredictiveCells()
    patternNZ = np.asarray(active_cell_indices)
    patternNZ = np.append(patternNZ, predictive_cell_indices)
    patternNZ = patternNZ.astype(np.int)
    patternNZ = list(set(patternNZ))

    result = c.compute(recordNum=i, patternNZ=patternNZ, classification={"bucketIdx": bucketIdx, "actValue": xi},learn=True, infer=True)
    topPredictions = sorted(zip(result[1], result["actualValues"]), reverse=True)[0]
    x_predict[i] = topPredictions[1]

tmp = np.asarray([np.arange(N-1), x_true, x_predict]).T
np.savetxt('log.csv', tmp, delimiter=',', header='time,true,predict', comments='')