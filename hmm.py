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
alpha = defaultdict(float)      #key j,i
trans = defaultdict(float)       #key j,i
t_ef = defaultdict(float)       #keyed by e_j,f_i
align = defaultdict(float)      #key i,i_prev, le
sys.stderr.write("initialization\n")
for (f, e) in bitext:
  le=len(e)
  lf=len(f)
  for (i, f_i) in enumerate(f):
      for (i_prev, f_i_prev) in enumerate(f):
        align[(i,i_prev,le)] = 1.0/(lf) ##+1)
  '''for (j,e_j) in enumerate(e):
      for (i,f_i) in enumerate(f):
        t_ef[(e_j,f_i)]=.25'''
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
    total_alpha = defaultdict(float)
    total_trans = defaultdict(float)
    state=defaultdict(float)
    for (n, (f, e)) in enumerate(bitext):
      le=len(e)
      lf=len(f)
      '''for i in range(len(f)):
        alpha[(0,i)]=1.0'''
      #alpha & beta
      for (j,e_j) in enumerate(e):
          total_alpha[j]=0
          for (i,f_i) in enumerate(f):      #What about null?
              if(j==0):
                  alpha[(n,j,i)]=t_ef[e_j,f_i]*align[(i,0,le)]
                  total_alpha[j] += alpha[(n,j,i)]
              else:
                for (i_prev,f_i_prev) in enumerate(f):
                  alpha[(n,j,i)] += (alpha[(n,j-1,i_prev)]/total_alpha[j-1])*align[(i,i_prev,le)]
                  total_alpha[j]+=alpha[(n,j,i)]
                alpha[(n,j,i)] *= t_ef[(e_j,f_i)]
                total_alpha[j]*=t_ef[(e_j,f_i)]
      #transition prob
      for (j,e_j) in enumerate(e):
          for (i,f_i) in enumerate(f):
              if(j==0):
                  trans[(j,i,0)] = t_ef[(e_j,f_i)]*align[(i,0,le)]*1##(alpha[(n,j-1,0)]/total_alpha[j-1])
                  total_trans[j]+= trans[(j,i,0)]
              else:
                  for (i_prev,f_i_prev) in enumerate(f):
                      trans[(j,i,i_prev)] = t_ef[(e_j,f_i)]*align[(i,i_prev,le)]*(alpha[(n,j-1,i_prev)]/total_alpha[j-1])
                      total_trans[j]+= trans[(j,i,i_prev)]
      '''#state prob
      for (j,e_j) in enumerate(e):
          for (i,f_i) in enumerate(f):
              if(j==0):
                  state[(j,i)]+=(trans[(j,i,0)]/total_trans[j])
              else:
                  for (i_prev,f_i_prev) in enumerate(f):
                      state[(j,i)]+=(trans[(j,i,i_prev)]/total_trans[j])'''
      #count collection
      for (j,e_j) in enumerate(e):
          for (i,f_i) in enumerate(f):
              count_ef[(e_j,f_i)] += (alpha[(n,j,i)]/total_alpha[(j)])#state[(j,i)]
              f_count[f_i]+= (alpha[(n,j,i)]/total_alpha[(j)]) #state[(j,i)]
              for (i_prev,f_i_prev) in enumerate(f):
                  count_a[(i,i_prev)] += (trans[(j,i,i_prev)]/total_trans[j])
                  total_a[i_prev]+= (trans[(j,i,i_prev)]/total_trans[j])
    for (f, e) in bitext:
        le = len(e)
        lf = len(f)
        for (j, e_j) in enumerate(e):
            for (i, f_i) in enumerate(f):
                t_ef[(e_j,f_i)] = count_ef[(e_j,f_i)]/f_count[f_i]
                if(j==0):
                  continue
                for (i_prev,f_i_prev) in enumerate(f):
                  align[(i,i_prev,le)]=count_a[(i,i_prev)]/total_a[i_prev]
for (n, (f, e)) in enumerate(bitext):
  le=len(e)
  lf=len(f)
  for (j,e_j) in enumerate(e):
    best_prob = 0
    best_i=0
    if(j==0):
        for (i, f_i) in enumerate(f):
            if(t_ef[(e_j,f_i)]>best_prob):
                best_i=i
                best_prob=t_ef[(e_j,f_i)]
    else:
        for (i, f_i) in enumerate(f):
            for (i_prev,f_i_prev) in enumerate(f):
                p_new = alpha[(n,j-1,i_prev)]*align[(i,i_prev,le)]*t_ef[(e_j,f_i)]
                if p_new > best_prob:
                    best_prob = p_new
                    best_i  = i
    sys.stdout.write("%i-%i " % (best_i,j))
  sys.stdout.write("\n")
sys.stderr.write(str(time.time()-start))
sys.stderr.write('\n')