''' Transform wiki corpus to conll2003 format: word, pos, chunk, IOB entity ''' 

 # global vars for checking whether a chunk should contiune or restart
NP, VP, PP = False, False, False
from sklearn.model_selection import train_test_split

def main():
	pos_tags = ['CC', 'CD', 'DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS'
				'NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM',
				'TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
	chunk_tags = ['NP', 'PP', 'VP'] #, 'ADVP', 'ADJP']

	''' entities: term, (IOB)genus, (IOB)hypernym '''
	global NP, VP, PP
	INPUT_DIRECTORY = "raw_data/"
	OUTPUT_DIRECTORY = "preprocessed_data/"
	instream = open(INPUT_DIRECTORY+'wiki_good.txt', 'r')

	# write data to corresponding file based on index
	out_train = open(OUTPUT_DIRECTORY+"train.txt", "w")
	out_train.write("-DOCSTART- -X- -X- O\n\n")
	out_test = open(OUTPUT_DIRECTORY+"test.txt", "w")
	out_test.write("-DOCSTART- -X- -X- O\n\n")
	out_valid = open(OUTPUT_DIRECTORY+"valid.txt", "w")
	out_valid.write("-DOCSTART- -X- -X- O\n\n")
	 # write all conll formated data to one file
	conll_output = open(OUTPUT_DIRECTORY+"wiki_conll.txt", "w")
	conll_output.write("-DOCSTART- -X- -X- O\n\n")

	lines = instream.readlines()
	indices = []
	sentences = []
	index = 0
	for i in range(1, len(lines), 2):
		indices.append(index)
		sentences.append(lines[i])
		index += 1

	x_train, x_test, y_train, y_test = train_test_split(sentences, indices, train_size=0.8)
	x_train, x_valid, y_train, y_valid =  train_test_split(x_train, y_train, train_size=0.9)
	y_train, y_test, y_valid = set(y_train), set(y_test), set(y_valid) # reduce later search complexity 

	for i in range(len(sentences)):
		if i in y_train:
			outstream = out_train
		elif i in y_test:
			outstream = out_test
		else:
			outstream = out_valid

		line = sentences[i]
		tokens = line.split("\t")
		term = tokens[0].split(":")[0]

		# replace terms like "Machine Learning" with "Machine-Learning"
		tmp = term.split(" ")
		if len(tmp) > 1:
			term = ""
			for w in tmp:
				term += "-" + w
			term = term[1:] # remove preceding "-"

		tokens = tokens[1:len(tokens)-1]
		# print(tokens)
		start_hyper, hyper, start_genus, genus = False, False, False, False
		
		for j in range(len(tokens)):
			token = tokens[j]
			# print(token)
			if token == "<VERB>" or token == "<REST>" or token == "":
				reset()
				continue
			elif token == "<HYPER>":
				start_hyper, hyper = True, True
				reset()
				continue
			elif token == "</HYPER>":
				hyper, genus = False, False
				start_hyper, start_genus = False, False
				reset()
				continue
			elif token == "<GENUS>": 
				start_genus, genus = True, True
				reset()
				continue

			l = token.split("_")

			# remove anomaly in dataset: no chunk or pos labelled for this word
			if len(l) == 1:
				continue
			
			# remove preceding blank for each token
			for i in range(len(l)):
				l[i] = l[i].strip()

			conll = "" # conll-2003 format

			# if "NP" in l or "PP" in l or "VP" in l:
			chunk_exists = False
			for i in l:
				if i in chunk_tags:
					chunk_exists = True
					break
			
			if chunk_exists:
				chunk, pos, word = l[0], l[1], l[2]
				# get chunk
				if chunk == "NP":
					VP, PP = False, False
					if not NP:
						NP = True
						chunk = "B-" + chunk
					else:
						chunk = "I-" + chunk
				elif chunk == "PP":
					NP, VP = False, False
					if not PP:
						PP = True
						chunk = "B-" + chunk
					else:
						chunk = "I-" + chunk
				elif chunk == "VP":
					NP, PP = False, False
					if not VP:
						VP = True
						chunk = "B-" + chunk
					else:
						chunk = "I-" + chunk
			
			else:
				# handle punctuations
				pos, word, chunk = l[0], l[1], "O"
				reset()

			# get entity
			entity = "O" # default
			if word == "TARGET":
				word = term
				entity = "B-TERM"
			elif start_hyper:
				entity = "B-HYPER"
				start_hyper = False
			elif hyper:
				entity = "I-HYPER"
			elif start_genus:
				entity = "B-GENUS"
				start_genus = False
			elif genus:
				entity = "I-GENUS"
			
			# replace a word/phrase containing blanks to a single word
			tmp = word.split(" ")
			if len(tmp) > 1:
				word = ""
				for w in tmp:
					word += "-" + w
				word = word[1:] # remove preceding "-"
			
			# remove noises
			if word != "" and pos != "":
				conll = "{} {} {} {}\n".format(word, pos, chunk, entity)

			conll_output.write(conll)
			outstream.write(conll)
		
		conll_output.write(". . O O\n\n")
		outstream.write(". . O O\n\n") # end of one definitional sentence
	
	# clear streams
	instream.close()
	out_train.close()
	out_test.close()
	out_valid.close()
			
def reset():
	global NP, VP, PP
	NP, VP, PP = False, False, False

if __name__ == "__main__":
	main()

		
