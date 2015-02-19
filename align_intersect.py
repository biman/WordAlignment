#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

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
align_fe = []#set()
align_ef = []#set()
sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
rf_count = defaultdict(int)
re_count = defaultdict(int)
ref_count = defaultdict(int)
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
      fe_count[(f_i,e_j)] += 1
  for e_j in set(e):
    e_count[e_j] += 1
  if n % 500 == 0:
    sys.stderr.write(".")

dice_fe = defaultdict(int)
for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
  dice_fe[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])
  if k % 5000 == 0:
    sys.stderr.write(".")
sys.stderr.write("\n")

for (n, (f, e)) in enumerate(bitext):
  for e_j in set(e):
      re_count[e_j] += 1
      for f_i in set(f):
         ref_count[(f_i,e_j)] += 1
  for f_i in set(f):
    rf_count[f_i] += 1
  if n % 500 == 0:
    sys.stderr.write(".")

dice_ef = defaultdict(int)
for (k, (f_i, e_j)) in enumerate(ref_count.keys()):
  dice_ef[(f_i,e_j)] = 2.0 * ref_count[(f_i, e_j)] / (rf_count[f_i] + re_count[e_j])
  if k % 5000 == 0:
    sys.stderr.write(".")
sys.stderr.write("\n")

for (n,(f, e)) in enumerate(bitext):
  align_fe.append(set())
  for (i, f_i) in enumerate(f):
    best_prob = 0
    best_j=0
    for (j, e_j) in enumerate(e):
      if dice_fe[(f_i,e_j)] > best_prob:
        best_prob = dice_fe[(f_i,e_j)]
        best_j  = j
      if dice_fe[(f_i,e_j)] == best_prob:
        if(abs(i-j)<abs(i-best_j)):
            best_prob = dice_fe[(f_i,e_j)]
            best_j  = j
    align_fe[n].add((i,best_j))

for (n,(f, e)) in enumerate(bitext):
  align_ef.append(set())
  for (j, e_j) in enumerate(e):
      best_prob = 0
      best_i=0
      for (i, f_i) in enumerate(f):
         if dice_ef[(f_i,e_j)] > best_prob:
            best_prob = dice_ef[(f_i,e_j)]
            best_i  = i
         if dice_ef[(f_i,e_j)] == best_prob:
            if(abs(j-i)<abs(j-best_i)):
                best_prob = dice_ef[(f_i,e_j)]
                best_i  = i
      align_ef[n].add((best_i,j))

for n in range(len(align_ef)):
    for (i,j) in align_ef[n]:
        #print (i,j)
        if((i,j) in align_fe[n]):
            sys.stdout.write("%i-%i " % (i,j))
    sys.stdout.write("\n")