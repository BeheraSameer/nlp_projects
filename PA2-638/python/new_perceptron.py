import sys
import getopt
import os
import math
from nltk.corpus import stopwords 
import numpy as np

class Perceptron:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Perceptron initialization"""
    self.numFolds     = 10  #Number of times the testing data is folded.
    self.words        = {}  #Dictionary for words in the bag of words.
    self.vocab_length = 0   #Total length of the calculated bag of words
    self.count_docs   = 0   #Count of total documents; just an iterator
    self.bag_of_words = []  #All word occurrences for all documents
    self.weights      = []  #Calculated feature weights

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    words = [word for word in words if word not in stopwords.words('english')]

    weight = 0.
    for w in list(words):
        w = w.lower()
        if w in self.words:
            weight = weight + float(self.weights[int(self.words[w])])
    print 'The calculated weight is: ', weight
    if weight > 0:
        return 'pos'
    else:
        return 'neg' 

    return 'pos'
  

  def addExample(self, klass, words, doc, iterations):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
     *
     Note:
         y_i is positive or negative depending on the document class. 1 for pos, -1 for neg.
         x_i is the individual feature
    """
    klass_int = -1
    if(klass == 'pos'):
        klass_int = 1
    occurrence = np.zeros(self.vocab_length)
    
    for i in range(0, iterations):
        sum_weights = 0.
        for w in set(words):
            w = w.lower()
            sum_weights += self.weights[int(self.words[w])]
            occurrence[int(self.words[w])] += 1
        #print 'The sum of weights for doc: ', doc, ' is: ', sum_weights
        for w in set(words):
            occur       = occurrence[int(self.words[w])]
            if np.sign(sum_weights * occur) != klass_int:
                self.weights[int(self.words[w])] += (klass_int - np.sign(sum_weights * occur)) * occur
    # Write code here

    pass
  
  def train(self, split, iterations):
      """
      * TODO 
      * iterates through data examples
      * TODO 
      * use weight averages instead of final iteration weights
      *
      * Personal notes:
      *  The initial for-loop iterates through the examples given as training data.
      *  From what it seems right now, train is a complete function.
      """
      
      np.random.shuffle(split.train)

      curr_word = 0
      for example in split.train:
          self.count_docs += 1
          for w in example.words:
              w = w.lower()
              if w not in self.words:
                  self.words[w] = curr_word
                  curr_word += 1
      self.vocab_length = len(self.words.keys())
      print 'The vocab length is: %d' % self.vocab_length + '\n'
      self.bag_of_words = np.zeros(self.vocab_length)
      self.weights      = np.zeros(self.vocab_length)
      
      ex_doc = 0

      for example in split.train:
          words = example.words
          self.addExample(example.klass, words, ex_doc, iterations)
          ex_doc += 1
      

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  

def test10Fold(args):
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
    
    
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    main()