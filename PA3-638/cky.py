import sys
import os
import copy
import string
from collections import defaultdict

class Parser:

    def __init__(self, grammar_path, sentences):
        if os.path.isfile(grammar_path):
            self.grammar_rule(grammar_path)
        if os.path.isfile(sentences):
            self.parseSents(sentences)

    def grammar_rule(self, grammar_path):
        #Dictionary of Grammar Rules
        grammar = defaultdict(float)
        #Set of Non-Terminals in the Grammar
        non_terminals = set()
        with open(grammar_path, 'rU') as f:
            for line in f:
                args = line.split()
                # print args
                non_terminals.add(args[0])
                if len(args) == 3:
                    grammar[(args[0], args[1])] = float(args[2])
                elif len(args) == 4:
                    grammar[(args[0], args[1], args[2])] = float(args[3])
        #print non_terminals
        #print grammar
        self.non_terminals = non_terminals
        self.grammar = grammar

    def parseSents(self, sentences):
        if os.path.isfile(sentences):
            with open(sentences) as input:
                for sentence in input:
                    self.parse(sentence)

    def parse(self, sentence):
        print("\n\nPROCESSING SENTENCE: " + sentence)
        sentence = sentence.split()
        word_count = len(sentence)
        scores = [[{} for j in range(word_count + 1)] for i in range(word_count + 1)]
        back_nodes = [[{} for j in range(word_count + 1)] for i in range(word_count + 1)]
        
        #CKY Implementation
        for i in range(word_count):
            word = sentence[i]
            print("\nSPAN:", word)
                
            for A in self.non_terminals:
                if (A, word) in self.grammar.keys():
                    scores[i][i + 1][A] = self.grammar[(A,word)]
                    print("P(" + A + " " + word + ") =", scores[i][i + 1][A])
                else:
                    scores[i][i + 1][A] = 0
                
            modified_nodes = set()
                
            #To Handle Unaries
            added = True
            while added:
                added = False
                #Not Modifying the Dictionary where Iteration is Going 
                cp_scores = copy.copy(scores[i][i + 1])
                for B in cp_scores:
                    for A in self.non_terminals:
                        if float(cp_scores[B]) > 0 and (A,B) in self.grammar.keys():
                            prob = float(self.grammar[(A,B)]) * float(cp_scores[B])
                            if prob > float(cp_scores[A]):
                                scores[i][i + 1][A] = prob
                                back_nodes[i][i + 1][A] = B
                                modified_nodes.add(A)
                                added = True
                
            for A in modified_nodes:
                print("P(" + A + ") =", scores[i][i + 1][A], "(BackPointer = (", back_nodes[i][i + 1][A], "))")
            
        #Implementing for Higher Layers
        for span in range(2, word_count + 1):
            for begin in range(word_count - span + 1):
                end = begin + span
                new_str = "\nSPAN: "
                for i in range(begin, end):
                    new_str += sentence[i] + " "
                print(new_str)
                    
                modified_up_nodes = set()
                    
                for split in range(begin + 1, end):
                    b_scores = scores[begin][split]
                    c_scores = scores[split][end]
                    for B in b_scores:
                        for A in self.non_terminals:
                            for C in c_scores:
                                #The Non-Terminal in A which has B as Left Child and C as Right Child
                                if (A, B, C) in self.grammar.keys():
                                    prob = float(b_scores[B]) * float(c_scores[C]) * float(self.grammar[(A,B,C)])
                                    if(prob <= 0):
                                        continue
                                    if not scores[begin][end] or A not in scores[begin][end] or prob > float(scores[begin][end][A]):
                                        scores[begin][end][A] = prob
                                        back_nodes[begin][end][A] = str(split) + ", " +  B + ", " + C
                                        modified_up_nodes.add(A)
                    
                #To Handle Unaries
                added = True
                while added:
                    added = False
                    #Not Modifying the Dictionary where Iteration is Going 
                    new_cp_scores = copy.copy(scores[begin][end])
                    for B in new_cp_scores:
                        for A in self.non_terminals:
                            if (A, B) in self.grammar.keys():
                                prob = float(self.grammar[(A,B)]) * float(new_cp_scores[B])
                                if not new_cp_scores or A not in new_cp_scores or prob > float(new_cp_scores[A]):
                                    scores[begin][end][A] = prob
                                    back_nodes[begin][end][A] = B
                                    modified_up_nodes.add(A)
                                    added = True
                    
                for A in modified_up_nodes:
                    print("P(" + A + ") =", scores[begin][end][A], "(BackPointer = (", back_nodes[begin][end][A], "))")
    
if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) != 3:
        print ("Wrong No of Arguments")
        print ("Usage: python3 cky.py grammar_rules.txt sents.txt")
        sys.exit()
    parser = Parser("grammar_rules.txt", "sents.txt")