''' Test if files for training follow the right format: eight fields are mandatory in each row '''

f = open("000_train.txt")
lines = f.readlines()
for line in lines:
	l = line.split(" ")
	if len(l) != 8:
		print(l)
		print(len(l))

f.close()