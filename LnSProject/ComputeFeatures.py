import os
import sys
import re
import numpy

fileString = str(sys.argv[1]);
file = open(fileString);
data = file.read();
file.close();
length = len(data);

count = 0;
# Convert data to documents
while True:
	count = count + 1;	
	doc_begin = data.find("~~~~~");
	doc_begin = doc_begin+6;
	data = data[doc_begin:length];
	doc_begin = 0;
	
	doc_end = data.find("~~~~~");
	
	if doc_end == -1:	
		document = data[doc_begin:length];
		doc_filename = 'articles_test/article' + str(count);
		file = open(doc_filename, 'w');
		file.write(document);
		file.close(); 
		print count;
		break;	
	
	doc_end = doc_end - 1;
	document = data[doc_begin:doc_end];

	doc_filename = 'articles_test/article' + str(count);
	file = open(doc_filename, 'w');
	file.write(document);
	file.close(); 
	
	data = data[doc_end + 1:length]; 	
	print count;

numDocuments = count;

# Initialize features
trigram_feat = numpy.zeros(numDocuments);
fourgram_feat = numpy.zeros(numDocuments);

# Compute perplexities using trigram LM
for i in range(1, numDocuments + 1):
	command = "echo \"perplexity -text articles_test/article" + str(i) + "\" | evallm -binary corpus3gram.binlm";	
	out = os.popen(command).read();
	
	for line in out.split("\n"):
		if line.startswith("Perplexity") == True:	
			x, y = map(float, re.findall(r'[+-]?\d+.\d+',line))
			trigram_feat[i-1] = x;

# Compute perplexities using 4gram LM
for i in range(1, numDocuments + 1):
	command = "echo \"perplexity -text articles_test/article" + str(i) + "\" | evallm -binary corpus4gram.binlm";	
	out = os.popen(command).read();
	
	for line in out.split("\n"):
		if line.startswith("Perplexity") == True:	
			x, y = map(float, re.findall(r'[+-]?\d+.\d+',line))
			fourgram_feat[i-1] = x;

# Standardize features
a = trigram_feat;
b = fourgram_feat;
a = (a - numpy.mean(a))/numpy.std(a);
b = (b - numpy.mean(b))/numpy.std(b);
trigram_feat = a;
fourgram_feat = b;

# Assign random labels
#file = open('developmentSetLabels.dat');
#data = file.read();
#file.close();
labels = numpy.zeros(numDocuments);
#ind = 0;
#for line in data.split("\n"):
#	if (ind < numDocuments):	
#		labels[ind] = int(line);
#		ind = ind + 1; 

# Generate ARFF file
file = open('articles_test.arff', 'w');
header = """@relation articles_test

@attribute Perplexity4gram real
@attribute Perplexity3gram real
@attribute label {0,1}

@data""";
file.write(header+"\n");
for i in range(0, numDocuments):
	featA = str(fourgram_feat[i]);
	featB = str(trigram_feat[i]);
	label = str(int(labels[i]));	
	file.write(featA+","+featB+","+label+"\n");
file.close();

		
			




