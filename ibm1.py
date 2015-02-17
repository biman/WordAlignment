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

#sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
e_file = [sentence.strip().split() for sentence in open(e_data).readlines()][:opts.num_sents]
f_file = [sentence.strip().split() for sentence in open(f_data).readlines()][:opts.num_sents]
e_count = defaultdict(int)
fe_count = defaultdict(int)
t_ef = defaultdict(float)
e_vocab = set()
f_vocab = set()
##each input word f may be translated with equal probability into any output word e.
for e in e_file:
    for e_j in e:
        e_vocab.add(e_j)
e_sz= len(e_vocab)
for f in f_file:
    for f_j in f:
        f_vocab.add(f_j)
for f in f_vocab:
    for e in e_vocab:   ## might only be required for (e,f) pairs that are in sentence pairs
        t_ef[(e,f)] = 1.0/e_sz
i=0
while(i<20):    ##NOT CONVERGED
    #print t_ef
    #sys.stderr.write(str(t_ef))
    flag=0
    count_ef=defaultdict(float)   #automatically initialized to 0
    f_count = defaultdict(float)  #automatically initialized to 0
    for (n, (f, e)) in enumerate(bitext):
      for e_j in set(e):
          e_count[e_j] = 0
          for f_i in set(f):
              e_count[e_j] += t_ef[(e_j,f_i)]
      #for e_j in set(e):
          for f_i in set(f):
              count_ef[(e_j,f_i)] += t_ef[(e_j,f_i)]/e_count[e_j]
              f_count[f_i] +=  t_ef[(e_j,f_i)]/e_count[e_j]
    #for (e,f) in count_ef.keys():
    for f in f_vocab:
        for e in e_vocab:
            if(abs(count_ef[(e,f)]/f_count[f]-t_ef[(e,f)])>0.01):  #if even one pair has big change, then iterate again
                flag=1
            t_ef[(e,f)] = count_ef[(e,f)]/f_count[f]
    #remove extra keys during first iteration
    '''if i==0:
        sys.stderr.write("deleting")
        count_keys = count_ef.keys()
        for (ek,fk) in t_ef.keys():
            if (ek,fk) not in count_keys:
                del t_ef[(ek,fk)]
                #sys.stderr.write(ek)
        sys.stderr.write("DONE")'''
    #if(flag==0):
    #    break
    i+=1
'''
    dice = defaultdict(int)
    for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
      dice[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])
      if k % 5000 == 0:
        sys.stderr.write(".")
    sys.stderr.write("\n")
'''
for (f, e) in bitext:
  for (i, f_i) in enumerate(f):
    best_prob = 0
    best_j=0
    for (j, e_j) in enumerate(e):
      if t_ef[(e_j,f_i)] >= best_prob:
        best_prob = t_ef[(e_j,f_i)]
        best_j  = j
    sys.stdout.write("%i-%i " % (i,best_j))
  sys.stdout.write("\n")
sys.stderr.write(str(time.time()-start))