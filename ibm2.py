#!/usr/bin/env python
import optparse
import sys
import time
from collections import defaultdict

start = time.time()
optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
#opts.threshold=0.4
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
e_file = [sentence.strip().split() for sentence in open(e_data).readlines()][:opts.num_sents]
f_file = [sentence.strip().split() for sentence in open(f_data).readlines()][:opts.num_sents]
e_count = defaultdict(int)
fe_count = defaultdict(int)
t_ef = defaultdict(float)
align = defaultdict(float)
'''e_vocab = set()
f_vocab = set()
##each input word f may be translated with equal probability into any output word e.
for e in e_file:
    for e_j in e:
        e_vocab.add(e_j)
e_sz= len(e_vocab)'''
#for f in f_file:
#    for f_j in f:
#        f_vocab.add(f_j)
#for f in f_vocab:
#    for e in e_vocab:   ## might only be required for (e,f) pairs that are in sentence pairs
sys.stderr.write("initialization\n")
for (f, e) in bitext:
  le=len(e)
  lf=len(f)
  for (j, e_j) in enumerate(e):
      for (i, f_i) in enumerate(f):
        #t_ef[(e_j,f_i)] = 1.0/e_sz      ##check if uniform or to be carried over from ibm1
        align[(i,j,le,lf)] = 1.0/(lf) ##+1)
#Get initial transition probabilities form ibm1- 5 iterations
ibm1_trans = open('t_ibm1')
t_ef = eval(ibm1_trans.readline())
#sys.stderr.write(t_ef)
sys.stderr.write("starting EM\n")
index=0
while(index<5):    ##NOT CONVERGED
    index+=1
    sys.stderr.write(str(index))
    sys.stderr.write("\n")
    count_ef=defaultdict(float)   #automatically initialized to 0 for each e,f
    f_count = defaultdict(float)  #automatically initialized to 0
    count_a = defaultdict(float)  #automatically init to 0 for each i,j,le,lf
    total_a = defaultdict(float)  #automatically init to 0 for each i,j,le,lf
    for (n, (f, e)) in enumerate(bitext):
      le=len(e)
      lf=len(f)
      for (j,e_j) in enumerate(e):
          e_count[e_j] = 0
          for (i,f_i) in enumerate(f):      #What about null?
              e_count[e_j] += t_ef[(e_j,f_i)]*align[(i,j,le,lf)]
      for (j,e_j) in enumerate(e):
          for (i,f_i) in enumerate(f):
              temp = t_ef[(e_j,f_i)]*align[(i,j,le,lf)]/e_count[e_j]
              count_ef[(e_j,f_i)] += temp
              f_count[f_i] +=  temp
              count_a[(i,j,le,lf)] += temp
              total_a[(j,le,lf)] += temp
    #for (e,f) in count_ef.keys():
    #for f in f_vocab:
    #    for e in e_vocab:
    for (f, e) in bitext:
        le = len(e)
        lf = len(f)
        for (j, e_j) in enumerate(e):
            for (i, f_i) in enumerate(f):
                t_ef[(e_j,f_i)] = count_ef[(e_j,f_i)]/f_count[f_i]
                align[(i,j,le,lf)] = count_a[(i,j,le,lf)]/total_a[(j,le,lf)]
for (f, e) in bitext:
  le=len(e)
  lf=len(f)
  for (i, f_i) in enumerate(f):
    best_prob = 0
    best_j=0
    for (j, e_j) in enumerate(e):
      if (t_ef[(e_j,f_i)]*align[(i,j,le,lf)]) >= best_prob:
        best_prob = t_ef[(e_j,f_i)]
        best_j  = j
    sys.stdout.write("%i-%i " % (i,best_j))
  sys.stdout.write("\n")
sys.stderr.write(str(time.time()-start))
sys.stderr.write('\n')