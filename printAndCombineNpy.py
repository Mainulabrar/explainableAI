import numpy as np
import re
import os
# '/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0ValueRep{i}step0'
pattern = rf'^0ValueRep(\d+)step0$'

Allvalues = np.zeros((18, 18)) 
for i in range(18):
	value = np.load(f'/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0ValueRep{i}step0.npy')
	Allvalues[i,:] = value

print(Allvalues)
np.savetxt('/data2/mainul/ExplainableAIResultsAllSteps/dataWithPlanscoreRun/0Allvalues.txt', Allvalues) 

