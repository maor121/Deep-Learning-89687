Train:

blistmTrain.py <repr> <trainFile> <modelFile> [<devFile>] [--ner] [--plot]

<repr>		 a , b, c or d
<trainFile>	 input tagged file
<modelFile> 	 model output file
[<devFile>]	 optional: evaluation file, tagged. Accuracy will be printed at end of run
[--ner]		 flag to evaluate accuracy in NER mode (requires: devFile)
[--plot]	 plot accuracy at end of run (requires: devFile)

Predict:

blistmPerdict.py <repr> <modelFile> <inputFile> <outputFile>

<repr>		 a , b, c or d
<inputFile>	 input file, tagged or untagged
<modelFile> 	 model file from train
<outputFile>	 output predicted file


Same info is availble by typing filename.py -h|--help

