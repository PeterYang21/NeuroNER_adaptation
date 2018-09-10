''' generate and plot confusion matrix for results of 
	deploying model trained on tp to testing texts that contain both tp and tn 

	tptn_test_label.txt: sentences (with labels) used for testing at the last epoch while training model fed with tptn sentences

	deploy_tp_on_tptn.txt: predicted entities of sentences above if they are labeled by model trained on only tp texts

	deply_result_label.txt: showing differences between true and predicted labels in format of "word, true entity, predicted entity"
	'''

from collections import defaultdict
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

''' generate 2 dicts to match prediction versus true label '''
def get_deploy_dic(text):
	deploy_dic = defaultdict(dict)
	key = None
	for i in range(len(text)):
		line = text[i].replace("\n","").split(" ")
		if line == [""]:
			continue
		elif "SENTID-" in line[0]:
			key = int(line[0].split("-")[1]) # sentence id as key
			continue
		word, pred = line[0], line[-1].replace("B-","").replace("I-","")
		deploy_dic[key][word] = pred
	
	return deploy_dic

def get_test_dic(text):
	test_dic = defaultdict(dict)
	key = 0
	for i in range(len(text)):
		line = text[i].replace("\n","").split(" ")
		if line == [""]:
			key += 1
			continue
		word, true = line[0], line[-2].replace("B-","").replace("I-","")
		test_dic[key][word] = true

	return test_dic

''' plot confusion matrix for 3 entities: term, definition, others '''
def plot_cf_matrix(y_pred, y_true):
	classes = ["DEF", "TERM", "O"]
	cm = confusion_matrix(y_true, y_pred, labels=classes)
	print(cm)

	p = plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap="Blues")
	plt.title("Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = "d"
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			plt.text(j, i, format(cm[i, j], fmt),
						horizontalalignment="center",
						color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
	p.savefig("test_w00tp_on_tptn.pdf", bbox_inches='tight')

def main():
	# match tags between 2 files
	labeled_file = "deploy_result_label.txt"
	outstream = open(labeled_file, "w")

	y_true, y_pred = [], []
	deploy_text = open("deploy_tp_on_tptn.txt","r").readlines()
	test_text = open("tptn_test_label.txt", "r").readlines()
	deploy_dic = get_deploy_dic(deploy_text)
	test_dic = get_test_dic(test_text)
	assert(test_dic.keys() == deploy_dic.keys())

	for key, d_dic in deploy_dic.items():
		t_dic = test_dic[key]
		for word, pred_entity in d_dic.items():
			if word in t_dic:
				true_entity = t_dic[word]
				y_true.append(true_entity)
				y_pred.append(pred_entity)
				outstream.write("{} {} {}\n".format(word, true_entity, pred_entity))
		outstream.write("\n")
		
	assert(len(y_pred) == len(y_true))
	plot_cf_matrix(y_pred, y_true)
	outstream.close()

if __name__ == "__main__":
	main()