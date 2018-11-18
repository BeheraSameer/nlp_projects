import sys
import getopt
import os
import math
import operator

class NaiveBayes:
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
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10
    
    # Total Count of Positive and Negative Words
    self.pos_cnt = 0
    self.neg_cnt = 0
    
    # Count of Positive and Negative Classes
    self.cnt_pos_clas = 0
    self.cnt_neg_clas = 0
    
    # My Set of All the Words
    self.my_dict = set()
    
    # Maintain a Dictionary for every Positive and Negative Word
    self.pos_wrd_cnt = {} 
    self.neg_wrd_cnt = {}

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  #
  # If any one of the FILTER_STOP_WORDS and BOOLEAN_NB flags is on, the
  # other one is meant to be off.

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    if self.FILTER_STOP_WORDS:
      words =  self.filterStopWords(words)
    
    # Write code here
    prob_word_given_pos = 0.0
    prob_word_given_neg = 0.0
    prob_pos_given_word = 0.0
    prob_neg_given_word = 0.0
    
    # To Calculate Total Dictionary Size
    dict_size = len(self.my_dict)
    
    # For Binarized Naive Bayes
    if self.BOOLEAN_NB:
      words = set(words)
    
    # Using Log and Laplace Smoothing For Easier Calculation of Probablilties
    for w in words:
      prob_word_given_pos += math.log(float((self.pos_wrd_cnt.get(w, 0) + 1) / float((self.pos_cnt + dict_size + 1))))
      prob_word_given_neg += math.log(float((self.neg_wrd_cnt.get(w, 0) + 1) / float((self.neg_cnt + dict_size + 1))))
    
    prob_pos_given_word += math.log((self.cnt_pos_clas / float(self.cnt_pos_clas + self.cnt_neg_clas))) + prob_word_given_pos
    prob_neg_given_word += math.log((self.cnt_neg_clas / float(self.cnt_pos_clas + self.cnt_neg_clas))) + prob_word_given_neg
    
    # Final Classification as Positive or Negative
    if prob_pos_given_word > prob_neg_given_word:
      return 'pos'
    else:
      return 'neg'
  

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the NaiveBayes class.
     * Returns nothing
    """
    
    # Write code here
    if self.FILTER_STOP_WORDS:
      words = self.filterStopWords(words)
    
    if klass == 'pos':
      self.cnt_pos_clas += 1
    else:
      self.cnt_neg_clas += 1
    
    # For Binarized Naive Bayes
    if self.BOOLEAN_NB:
      words = set(words)
    
    # Word Count Update
    for word in words:
      if klass == 'pos':
        self.pos_wrd_cnt[word] = self.pos_wrd_cnt.get(word, 0) + 1
        self.pos_cnt += 1
      elif klass == 'neg':
        self.neg_wrd_cnt[word] = self.neg_wrd_cnt.get(word, 0) + 1
        self.neg_cnt += 1 
      self.my_dict.add(word)

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

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a list of TrainSplits corresponding to the cross validation splits."""
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
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB):
  nb = NaiveBayes()
  splits = nb.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)
  
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
    
    
def classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB, trainDir, testDir):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testSplit = classifier.trainSplit(testDir)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy


def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True
  
  if len(args) == 2:
    classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB,  args[0], args[1])
  elif len(args) == 1:
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB)

if __name__ == "__main__":
    main()
