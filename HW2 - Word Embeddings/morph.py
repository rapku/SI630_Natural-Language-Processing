import pickle, re

uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))


morph1code = re.compile(r'(\w+)ly$')
morph2code = re.compile(r'^un(\w+)')
morph3code = re.compile(r'(\w+)ing$')
morph4code = re.compile(r'(\w+)ion$')
morph5code = re.compile(r'(\w+)ity$')
morph1, morph2, morph3, morph4, morph5, trainmorph, devmorph, testmorph = ([] for i in range(8))
allmorph = [morph1, morph2, morph3, morph4, morph5]


for x in uniqueWords:
	a = morph1code.match(x)
	if a != None:
		if a.group(1) in uniqueWords and len(a.group(1)) >= 3:
			morph1.append((a.group(1), a.string))

for x in uniqueWords:
	a = morph2code.match(x)
	if a != None:
		if a.group(1) in uniqueWords and len(a.group(1)) >= 3:
			morph2.append((a.group(1), a.string))

for x in uniqueWords:
	a = morph3code.match(x)
	if a != None:
		if a.group(1) in uniqueWords and len(a.group(1)) >= 3:
			morph3.append((a.group(1), a.string))

for x in uniqueWords:
	a = morph4code.match(x)
	if a != None:
		if a.group(1) in uniqueWords and len(a.group(1)) >= 3:
			morph4.append((a.group(1), a.string))

for x in uniqueWords:
	a = morph5code.match(x)
	if a != None:
		if a.group(1) in uniqueWords and len(a.group(1)) >= 3:
			morph5.append((a.group(1), a.string))
del morph5[18]
allmorph = [morph1, morph2, morph3, morph4, morph5]

for morph in allmorph:
	for x in range(16):
		trainmorph.append(morph[x])
	for x in range(16, 18):
		devmorph.append(morph[x])
	for x in range(18, 20):
		testmorph.append(morph[x])

print(trainmorph)