import numpy as np

patient = '098'
i = 15

IGvalues = np.loadtxt(f'/data2/mainul/explainableAIResultsMultiplePatientCases/PaperInit/{patient}attribution2ndRep{i}.txt')

print(IGvalues)
print(np.sum(IGvalues))	    