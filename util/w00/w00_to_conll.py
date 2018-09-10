''' Transform W00 corpus to conll2003 format: word, pos, chunk, IOB entity ''' 

 # global vars for checking whether a chunk should contiune or restart
NP, VP, PP = False, False, False
from sklearn.model_selection import train_test_split
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-data", dest="data", help="include only tp or tp+tn", default="tp")
	args = parser.parse_args()

	# pos_tags = ['CC', 'CD', 'DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS'
	# 			'NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM',
	# 			'TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
	
	global NP, VP, PP
	INPUT_DIRECTORY = "raw_data/"
	OUTPUT_DIRECTORY = args.data + "/"

	meta_stream = open(INPUT_DIRECTORY+"annotated.meta",'r').readlines()
	entity_stream  = open(INPUT_DIRECTORY+"annotated.tag",'r').readlines()
	pos_stream  = open(INPUT_DIRECTORY+"annotated.pos",'r').readlines()
	chunk_stream  = open(INPUT_DIRECTORY+"annotated.chunk",'r').readlines()
	word_stream  = open(INPUT_DIRECTORY+"annotated.word",'r').readlines()

	# write data to corresponding file based on index
	out_train = open(OUTPUT_DIRECTORY+"train.txt", "w")
	out_train.write("-DOCSTART- -X- -X- O\n\n")
	out_test = open(OUTPUT_DIRECTORY+"test.txt", "w")
	out_test.write("-DOCSTART- -X- -X- O\n\n")
	out_valid = open(OUTPUT_DIRECTORY+"valid.txt", "w")
	out_valid.write("-DOCSTART- -X- -X- O\n\n")
	 # write all conll formated data to one file
	conll_output = open(OUTPUT_DIRECTORY+"w00_conll.txt", "w")
	conll_output.write("-DOCSTART- -X- -X- O\n\n")

	# write data to corresponding file based on index
	indices, sentences = [], []
	
	# maximum tp in w00 is 730
	# only tp: #training: 410 #validation: 137 #testing: 183
	# tp + tn(730): #training: 821 #validation: 274 #testing: 365
	tp, tn = 730, 0 

	for i in range(len(meta_stream)):
		if meta_stream[i][0] == "1":
			indices.append(i)
			sentences.append(word_stream[i])
		elif args.data == "tptn":
			if tn < tp:
				indices.append(i)
				sentences.append(word_stream[i])
				tn += 1

	x_train, x_test, y_train, y_test = train_test_split(sentences, indices, test_size=0.2)
	x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
	y_train, y_test, y_valid = set(y_train), set(y_test), set(y_valid) # reduce later search complexity 
	print("#training: {} #validation: {} #testing: {}".format(len(y_train), len(y_valid), len(y_test)))
	for i in indices:
		if args.data == "tp" and meta_stream[i][0] == "0":
			continue
		if i in y_train:
			outstream = out_train
		elif i in y_test:
			outstream = out_test
		else:
			outstream = out_valid

		TERM, DEF = False, False
		words = word_stream[i].strip("\n").split(" ")
		entities = entity_stream[i].strip("\n").split(" ")
		poss = pos_stream[i].strip("\n").split(" ")
		chunks = chunk_stream[i].strip("\n").split(" ")
		reset()
		
		for j in range(len(words)):
			word, pos, chunk, entity = words[j], poss[j], chunks[j], entities[j]
			chunk = chunk[:-1]
			if chunk != "NP" and chunk!= "VP" and chunk != "PP":
				chunk = "O"
				reset()
			else:
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
			
			# handle entity
			if entity == "TERM":
				DEF = False
				if not TERM:
					TERM = True
					entity = "B-" + entity
				else:
					entity = "I-" + entity
			elif entity == "DEF":
				TERM = False
				if not DEF:
					DEF = True
					entity = "B-" + entity
				else:
					entity = "I-" + entity
			else:
				# entity is "O"
				DEF, TERM = False, False

			conll = "{} {} {} {}\n".format(word, pos, chunk, entity)
			conll_output.write(conll)
			outstream.write(conll)

		conll_output.write("\n")
		outstream.write("\n")
	conll_output.close()
	out_train.close()
	out_test.close()
	out_valid.close()
			
def reset():
	global NP, VP, PP
	NP, VP, PP = False, False, False

if __name__ == "__main__":
	main()