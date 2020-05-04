from sklearn.ensemble import RandomForestClassifier
import numpy as np

trainlist = []
testlist=[]
class Domain:
	def __init__(self,_name,_label):
		self.name = _name
		self.label = _label
		self.namelength = len(_name)
		num_digit = 0
		num_alpha = 0
		for i in _name:
			if(i.isdigit()):
				num_digit = num_digit + 1
			else :
				num_alpha = num_alpha + 1
		self.num_digit = num_digit
		self.entropy = num_digit/num_alpha
	def returnData(self):
		return [self.namelength, self.num_digit, self.entropy]

	def returnLabel(self):
		if self.label == "dga":
			return 0
		else:
			return 1
	def changeLabel(self):
		self.label="notdga"
	def returnResult(self):
		return self.name,self.label
def initData(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line == "":
				continue
			tokens = line.split(",")
			name = tokens[0]
			label = tokens[1]
			trainlist.append(Domain(name,label))
def initTest(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			tokens = line.split(",")
			name = tokens[0]
			label = "dga"
			testlist.append(Domain(name,label))
def main():
	initData("train.txt")
	initTest("test.txt")
	featureMatrix = []
	labelList = []
	for item in trainlist:
		featureMatrix.append(item.returnData())
		labelList.append(item.returnLabel())

	clf = RandomForestClassifier(random_state=0)
	clf.fit(featureMatrix,labelList)
	with open("result.txt",'w') as fp:
		for i in testlist:
			if clf.predict([i.returnData()]) [0] == 1:
				i.changeLabel()
			a,b = i.returnResult()
			fp.write(a +" " + b + "\n")
if __name__ == '__main__':
	main()
