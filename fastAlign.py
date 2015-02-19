#!/usr/bin/env python
##Using Diagonal Element
import optparse
import sys
import time
import math
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
#h= defaultdict(float)
align = defaultdict(float)
_lambda=1
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
  r=math.exp(-_lambda/lf)
  for (j, e_j) in enumerate(e):
      for (i, f_i) in enumerate(f):
        #t_ef[(e_j,f_i)] = 1.0/e_sz      ##check if uniform or to be carried over from ibm1
        i_up = (j*lf)/le
        i_down = i_up+1
        h_up = abs((i_up/lf)-(j/le))
        h_down = abs((i_down/lf)-(j/le))
        h= abs((i/lf)-(j/le))
        z_up=math.exp(_lambda*h_up)*(1-pow(r,(i_up+1)))/(1-r)
        z_down=math.exp(_lambda*h_down)*(1-pow(r,(lf-i_down)))/(1-r)
        align[(i,j,le,lf)] = math.exp(_lambda*h)/(z_up+z_down) #1.0/(lf+1)
        #h[(i,j,le,lf)] = abs((i/lf)-(j/le))
#Get initial transition probabilities form ibm1- 5 iterations
ibm1_trans = open('t_ibm1')
t_ef = eval(ibm1_trans.readline())
#sys.stderr.write(t_ef)
sys.stderr.write("starting EM\n")
index=0
while(index<5):    ##NOT CONVERGED
    #print t_ef
    index+=1
    sys.stderr.write(str(index))#str(t_ef))
    #sys.stderr.write(str(t_ef))
    sys.stderr.write("\n")
    flag=0
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
      #for (j,e_j) in enumerate(e):
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
    #s=0
    #for (i,j,le,lf) in align.keys():
        #s+=align[(i,j,le,lf)]*abs((i/lf)-(j/le))
    #gradient_iter=1
    #while(gradient_iter<=5):
        #gradient_iter+=1
        ##s_diff=
        #gradient = s-
for (f, e) in bitext:
  le=len(e)
  lf=len(f)
  for (i, f_i) in enumerate(f):
    best_prob = 0
    best_j=0
    for (j, e_j) in enumerate(e):
      if t_ef[(e_j,f_i)]*align[(i,j,le,lf)] >= best_prob:
        best_prob = t_ef[(e_j,f_i)]
        best_j  = j
    sys.stdout.write("%i-%i " % (i,best_j))
  sys.stdout.write("\n")
sys.stderr.write(str(time.time()-start))
sys.stderr.write('\n')

