# f = open("tptn_test.txt", "r")
# lines = f.readlines()
# outstream = open("tptn_test_deploy.txt", "w")
# id = 0
# s = "SENTID-" + str(id)
# for line in lines:
# 	line = line.split(" ")
# 	if line[-1] != "\n":
# 		s += " " + line[0]
# 	else:
# 		outstream.write(s.strip() + "\n")
# 		id += 1
# 		s = "SENTID-" + str(id)
	
f = open("tptn_test.txt", "r")
lines = f.readlines()
outstream = open("tptn_test_deploy.txt", "w")

for line in lines:
	line = line.split(" ")
	if line[-1] != "\n":
		outstream.write(" " + line[0])
	else:
		outstream.write("\n")

f.close()
outstream.close()

