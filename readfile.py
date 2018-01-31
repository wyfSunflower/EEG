#!/usr/bin/python
with open('./exp1.dat', 'r') as f:
	begin = 0
	for line in f.readlines():
		begin = begin + 1;
		channel = []
		if begin > 14 :#valid data is after line 14
			#electrode 0 is reference 
			channel.append([float(line.split()[1]) - float(line.split()[0]),
				float(line.split()[2]) - float(line.split()[0]),
				float(line.split()[3]) - float(line.split()[0])])
			print(channel)


#make fourier transform, wavlet or cosin transform on each sample to be sparse 

#split every 1000*3 as one sample

#make matrix decomposition on each sample to decrease dimension, if it can be hundreds*3 or tens*3 will be better

#save the new sample into one file and input them into model to train

#before train need shuffle all samples and divide them into train set80%, test set10%, dev set 10%

#for svm model try several kernel

#use soft margin svm have a try

#use one dimension cnn to have a try, use different size of kernel and stride

#use semisupervise unsupervise learning ,reafinforce learning such as q learning try
