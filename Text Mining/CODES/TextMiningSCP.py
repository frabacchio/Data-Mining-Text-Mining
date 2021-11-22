import os
import numpy as np

specialchar = [',', ':', '.', ';', '"', '?', '!', '(', ')', '%', '$ ', '=',"[","]","{","}"]

corpus_directory='corpus2mw'
#corpus_directory='testi'
def token(w):
    res = []
    count = specialchar.count(w[0]) + specialchar.count(w[-1])

    if count == 1:
        if specialchar.count(w[0]):
            res.append(w[0])
            res.append(w[1:])
        if specialchar.count(w[-1]):
            res.append(w[:-1])
            res.append(w[-1])
    if count == 2:
        res.append(w[0])
        res.append(w[1:-1])
        res.append(w[-1])

    return res


corpus = []
texts_names = os.listdir('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/'+corpus_directory)

with open('doc_order.txt', 'w') as f:
    count=0
    for line in texts_names:
        f.write('doc'+str(count)+": ")
        f.write(line)
        f.write('\n')
        count+=1

print('start reading corpus')
for text in texts_names:

    allwords_in_a_text = []
    with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/'+corpus_directory+'/' + text, 'r', errors='ignore') as file:

        for line in file:
            for word in line.split():
                aux = token(word)
                if len(aux) == 2:
                    allwords_in_a_text.append(aux[0])
                    allwords_in_a_text.append(aux[1])
                elif len(aux) == 3:
                    allwords_in_a_text.append(aux[0])
                    allwords_in_a_text.append(aux[1])
                    allwords_in_a_text.append(aux[2])

                else:
                    allwords_in_a_text.append(word)

    corpus.append(allwords_in_a_text)

print('finished reading corpus')

def list_to_str(word):
    res = ""
    for a in range(len(word) - 1):
        res += word[a] + " "
    res += word[-1]
    return res


def str_to_list(s):
    res = []
    count = 0
    for i in range(len(s)):
        if (s[i] == " "):
            res.append(s[i - count:i])
            count = 0
        else:
            count += 1
    i = len(s)
    res.append(s[i - count:i])
    return res




def create_list_of_dict_global(max_ngrams,corpus1):
  list_dict=[]
  for k in range(max_ngrams):
    list_dict.append({})

  for doc_number in range(len(corpus1)):
      for i in range(len(corpus[doc_number])):
        for ngrams in range(max_ngrams):
           if (i+ngrams+1)<=len(corpus1[doc_number]):
              if (list_to_str(corpus[doc_number][i:i+ngrams+1]) in list_dict[ngrams].keys()):
                  list_dict[ngrams][list_to_str(corpus1[doc_number][i:i+ngrams+1])][0]+=1
                  if list_dict[ngrams][list_to_str(corpus1[doc_number][i:i+ngrams+1])][-2]!=doc_number:
                    list_dict[ngrams][list_to_str(corpus1[doc_number][i:i + ngrams + 1])].append(doc_number)
                    list_dict[ngrams][list_to_str(corpus1[doc_number][i:i + ngrams + 1])].append(1)
                  else:
                    list_dict[ngrams][list_to_str(corpus1[doc_number][i:i + ngrams + 1])][-1]+=1

              else:
                  list_dict[ngrams][list_to_str(corpus1[doc_number][i:i+ngrams+1])]=[]
                  list_dict[ngrams][list_to_str(corpus1[doc_number][i:i+ngrams+1])].append(1)
                  list_dict[ngrams][list_to_str(corpus1[doc_number][i:i + ngrams + 1])].append(doc_number)
                  list_dict[ngrams][list_to_str(corpus1[doc_number][i:i + ngrams + 1])].append(1)

  return list_dict

def find_indices_ngram_doc(strngram,docnum):
    indices=[]
    doc=corpus[docnum]
    listngram=str_to_list(strngram)
    for i in range(0,len(doc)-len(listngram)+1):
        if listngram[0]==doc[i]:
                flag=1
                locount=1
                while flag and locount+i<len(doc) and locount<len(listngram):
                    if doc[locount+i]!=listngram[locount]:
                        flag=0
                    else:
                        locount+=1
                if locount==len(listngram):
                    indices.append(i)

    return indices





print('start creating final_dict')
final_dict=create_list_of_dict_global(8,corpus)
print('finished creating final_dict')


def create_glue_and_tfidfmod_and_probs(final_dict1,gluename,corpus):
    num_total_docs=len(corpus)
    final_dict2 = []
    final_dict3 = []
    #final_dict4 = []
    if gluename=='Dice':
        for ngram in range(1, len(final_dict1)):
            g = dict(final_dict1[ngram - 1])
            g2 = dict(final_dict1[ngram - 1])
            #g3= dict(final_dict1[ngram - 1])
            for keys, value in final_dict1[ngram - 1].items():
                if not(ngram==1 and len(keys)<3):
                    tfidf_mod=[]
                    probs=[]
                    #positions=[]
                    num_non_zero_doc=(len(value)-1)/2
                    for doc_num in range(1,len(value),2):
                        vec_len_words=[]
                        for j in str_to_list(keys):
                            vec_len_words.append(len(j))
                        tfidf_mod.append(value[doc_num])
                        tfidf_mod.append(np.mean(vec_len_words)*value[doc_num+1]*np.log(num_total_docs/num_non_zero_doc)/len(corpus[value[doc_num]]))
                        probs.append(value[doc_num])
                        probs.append(value[doc_num+1]/len(corpus[value[doc_num]]))
                        #positions.append(value[doc_num])
                        #positions.append(find_indices_ngram_doc(keys,value[doc_num]))

                    medprob=sum(probs[1::2])/num_total_docs
                    probs[1::2]=[l1-medprob for l1 in probs[1::2]]

                    num=value[0]
                    g[keys]=[]
                    g2[keys]=[]
                    #g3[keys]=[]
                    if ngram!=1:
                        key = str_to_list(keys)
                        somma = 0
                        for i in range(len(key) - 1):
                            f1 = final_dict1[i][list_to_str(key[:i + 1])][0]
                            f2 = final_dict1[ngram - i - 2][list_to_str(key[i + 1:])][0]
                            somma += (f1 + f2) / (ngram - 1)
                        g[keys].append((2 * num) / somma)
                        g2[keys].append((2 * num) / somma)
                       # g3[keys].append((2 * num) / somma)
                    for val in tfidf_mod:
                        g[keys].append(val)
                    for val2 in probs:
                        g2[keys].append(val2)
                    #for val3 in positions:
                        #g3[keys].append(val3)
                else:
                    g.pop(keys)
                    g2.pop(keys)
                    #g3.pop(keys)

            final_dict2.append(g)
            final_dict3.append(g2)

    elif gluename == 'SCP':
        for ngram in range(1, len(final_dict1)):
            g = dict(final_dict1[ngram - 1])
            g2 = dict(final_dict1[ngram - 1])
            # g3= dict(final_dict1[ngram - 1])
            for keys, value in final_dict1[ngram - 1].items():
                if not (ngram == 1 and len(keys) < 3):
                    tfidf_mod = []
                    probs = []
                    # positions=[]
                    num_non_zero_doc = (len(value) - 1) / 2
                    for doc_num in range(1, len(value), 2):
                        vec_len_words = []
                        for j in str_to_list(keys):
                            vec_len_words.append(len(j))
                        tfidf_mod.append(value[doc_num])
                        tfidf_mod.append(np.mean(vec_len_words) * value[doc_num + 1] * np.log(
                            num_total_docs / num_non_zero_doc) / len(corpus[value[doc_num]]))
                        probs.append(value[doc_num])
                        probs.append(value[doc_num + 1] / len(corpus[value[doc_num]]))
                        # positions.append(value[doc_num])
                        # positions.append(find_indices_ngram_doc(keys,value[doc_num]))

                    medprob = sum(probs[1::2]) / num_total_docs
                    probs[1::2] = [l1 - medprob for l1 in probs[1::2]]

                    num = value[0]
                    g[keys] = []
                    g2[keys] = []
                    # g3[keys]=[]
                    if ngram != 1:
                        key = str_to_list(keys)
                        somma = 0
                        for i in range(len(key) - 1):
                            f1 = final_dict1[i][list_to_str(key[:i + 1])][0]
                            f2 = final_dict1[ngram - i - 2][list_to_str(key[i + 1:])][0]
                            somma += (f1*f2) / (ngram - 1)
                        g[keys].append((num**2) / somma)
                        g2[keys].append((num**2) / somma)
                    # g3[keys].append((2 * num) / somma)
                    for val in tfidf_mod:
                        g[keys].append(val)
                    for val2 in probs:
                        g2[keys].append(val2)
                    # for val3 in positions:
                    # g3[keys].append(val3)
                else:
                    g.pop(keys)
                    g2.pop(keys)
                    # g3.pop(keys)

            final_dict2.append(g)
            final_dict3.append(g2)
            #final_dict4.append(g3)

    elif gluename == 'MI':
        for ngram in range(1, len(final_dict1)):
            g = dict(final_dict1[ngram - 1])
            g2 = dict(final_dict1[ngram - 1])
            # g3= dict(final_dict1[ngram - 1])
            for keys, value in final_dict1[ngram - 1].items():
                if not (ngram == 1 and len(keys) < 3):
                    tfidf_mod = []
                    probs = []
                    # positions=[]
                    num_non_zero_doc = (len(value) - 1) / 2
                    for doc_num in range(1, len(value), 2):
                        vec_len_words = []
                        for j in str_to_list(keys):
                            vec_len_words.append(len(j))
                        tfidf_mod.append(value[doc_num])
                        tfidf_mod.append(np.mean(vec_len_words) * value[doc_num + 1] * np.log(
                            num_total_docs / num_non_zero_doc) / len(corpus[value[doc_num]]))
                        probs.append(value[doc_num])
                        probs.append(value[doc_num + 1] / len(corpus[value[doc_num]]))
                        # positions.append(value[doc_num])
                        # positions.append(find_indices_ngram_doc(keys,value[doc_num]))

                    medprob = sum(probs[1::2]) / num_total_docs
                    probs[1::2] = [l1 - medprob for l1 in probs[1::2]]

                    num = value[0]
                    g[keys] = []
                    g2[keys] = []
                    # g3[keys]=[]
                    if ngram != 1:
                        key = str_to_list(keys)
                        somma = 0
                        for i in range(len(key) - 1):
                            f1 = final_dict1[i][list_to_str(key[:i + 1])][0]
                            f2 = final_dict1[ngram - i - 2][list_to_str(key[i + 1:])][0]
                            somma += (f1*f2) / (ngram - 1)
                        g[keys].append(np.log(num / somma))
                        g2[keys].append(np.log(num / somma))
                    # g3[keys].append((2 * num) / somma)
                    for val in tfidf_mod:
                        g[keys].append(val)
                    for val2 in probs:
                        g2[keys].append(val2)
                    # for val3 in positions:
                    # g3[keys].append(val3)
                else:
                    g.pop(keys)
                    g2.pop(keys)
                    # g3.pop(keys)

            final_dict2.append(g)
            final_dict3.append(g2)
            #final_dict4.append(g3)

    return final_dict2,final_dict3

print('start creating glue and tfidfmod_and_probs')
glues,probs= create_glue_and_tfidfmod_and_probs(final_dict,'SCP',corpus)
print('finished creating glue and tfidfmod')

print('start creating fathers')
fathers = []
for ngram in range(1, 8):
    father = dict(final_dict[ngram - 1])
    for key, value in father.items():
        father[key] = []

    for keys, value in final_dict[ngram].items():
        key = str_to_list(keys)
        subkey1 = list_to_str(key[1:])
        subkey2 = list_to_str(key[:-1])
        father[subkey1].append(keys)
        father[subkey2].append(keys)

    fathers.append(father)

print('finished creating fathers')

def pi(rel,num_doc):
    for i in range(1,len(final_dict[len(str_to_list(rel))-1][rel])):
        if(final_dict[len(str_to_list(rel))-1][rel][i]==num_doc):
            if(i%2==1):
               return final_dict[len(str_to_list(rel))-1][rel][i+1]/(len(corpus[i%2-1]))
    return 0

def find_RE(final_dict1,w,w2):
    re_tfidf = {}
    re_probs= {}
    #re_positions={}
    for k in range(len(w)-1):
        if (k > 1):
            for keys, value in w[k].items():
                flag = 1
                key = str_to_list(keys)
                subkey1 = list_to_str(key[1:])
                subkey2 = list_to_str(key[:-1])
                subv1 = w[k - 1][subkey1][0]
                subv2 = w[k - 1][subkey2][0]
                subvalues = [subv1, subv2]
                supvalues = []
                for fath in fathers[k][keys]:
                    c = w[k + 1][fath][0]
                    supvalues.append(c)
                for subv in subvalues:
                    for supv in supvalues:
                        # if value[0] <= (subv + supv) / 2:
                        #     flag = 0
                        if value[0]<=subv or value[0]<=supv:
                            flag=0
                if flag:
                    re_tfidf[keys]=value[1:]
                    re_probs[keys]=w2[k][keys][1:]
                    #re_positions[keys] = w3[k][keys][1:]
        elif k==1:
            for keys, value in w[k].items():
                flag = 1
                supvalues = []
                for fath in fathers[k][keys]:
                    c = w[k + 1][fath][0]
                    supvalues.append(c)

                for supv in supvalues:
                    if value[0] <=  supv :
                        flag = 0
                if flag:
                    re_tfidf[keys]=value[1:]
                    re_probs[keys] = w2[k][keys][1:]
                    #re_positions[keys] = w3[k][keys][1:]



    return re_tfidf,re_probs


print('start finding RE')
retfidf,reprobs= find_RE(final_dict, glues,probs)
print('finished finding RE')

with open('RE_SCP_corpus2mw_unfiltered.txt', 'w') as f:
    for key,value in reprobs.items():
        f.write(key)
        f.write('\n')

def tfidf_unigrams(w,w2): #pass to the function w[0]
    unitfidf={}
    uniprobs={}
    #unipositions={}
    for key,value in w.items():
        unitfidf[key]=w[key]
        uniprobs[key]=w2[key]
        #unipositions[key]=w3[key]
    return unitfidf,uniprobs

print('start creating tfidf_uni')
unitfidf,uniprobs = tfidf_unigrams(glues[0],probs[0])
print('finished creating tfidf_uni')

def is_okay(rel):
    for i in range(len(rel)):
        if (rel[i] in specialchar):
            return False
    return True


def is_okay2(rel):
    if (final_dict[len(str_to_list(rel)) - 1][rel][0] > 1):
        return True
    else:
        return False

vect3=["in","as","for","to","it","if","who","with","when","what","where","away","below","behind","by","of","on","and","that","those","these","this","is","an","a","per"]
vect4=["in","as","for","to","it","if","who","with","when","what","where","by","of","on","and","that","those","these","this","is","an","a","per"]
def is_okay3(re):
  re=str_to_list(re)
  if (len(re)<3):
    for i in range(len(re)):
      if (re[i] in vect3):
        return False
  return True

def is_okay4(re):
    re = str_to_list(re)
    if re[-1] in vect4:
        return False

    return True


retfidf_okay = {}
unitfidf_okay={}
reprobs_okay = {}
uniprobs_okay={}
# repositions_okay = {}
# unipositions_okay={}
for key,value in retfidf.items():
    if (is_okay(key) and is_okay2(key) and is_okay3(key) and is_okay4(key)):
        retfidf_okay[key]=value
        reprobs_okay[key]=reprobs[key]
        #repositions_okay[key] = repositions[key]
for key,value in unitfidf.items():
    if (is_okay(key) and is_okay2(key) and is_okay3(key) and is_okay4(key)):
        unitfidf_okay[key]=value
        uniprobs_okay[key] = uniprobs[key]
        #unipositions_okay[key] = unipositions[key]

print(str(len(retfidf_okay))+'RE_okay found')
with open('RE_SCP_corpus2mw.txt', 'w') as f:
    for key,value in reprobs_okay.items():
        f.write(key)
        f.write('\n')

np.random.seed(123456789)
re_okay=list(retfidf_okay.keys())
re_okay=np.array(re_okay)
indexes_random=np.random.choice(len(re_okay),100)
re_okay_random_100=re_okay[indexes_random]

with open('RE_SCP_random_100.txt', 'w') as f:
    for line in re_okay_random_100:
        f.write(line)
        f.write('\n')



def findExplicit_Keywords(corpus,RE,uni,numberuniKeywords,numbermultiKeywords):

    dict_uni_docs={}
    dict_re_docs = {}
    for k in range(len(corpus)):
       dict_uni_docs['doc'+str(k)]=[]
       dict_uni_docs['tfidf'+str(k)]=[]
       dict_re_docs['doc'+str(k)]=[]
       dict_re_docs['tfidf'+str(k)]=[]

    for keyuni,valueuni in uni.items():
      count=1
      for doc in valueuni[::2]:
          dict_uni_docs['doc'+str(doc)].append(keyuni)
          dict_uni_docs['tfidf' +str(doc)].append(valueuni[count])
          count+=2

    dict_uni_docs_final = {}
    for k in range(len(corpus)):
       dict_uni_docs_final['doc'+str(k)]=[x for _, x in sorted(zip(dict_uni_docs['tfidf'+str(k)],dict_uni_docs['doc'+str(k)]),reverse=True)]
       if len(dict_uni_docs_final['doc'+str(k)])>=numberuniKeywords:
           dict_uni_docs_final['doc'+str(k)]= dict_uni_docs_final['doc'+str(k)][:numberuniKeywords]

    for key, value in RE.items():
        count=1
        for doc in value[::2]:
            dict_re_docs['doc' + str(int(doc))].append(key)
            dict_re_docs['tfidf' + str(int(doc))].append(value[count])
            count+=2

    dict_re_docs_final={}
    for k in range(len(corpus)):
       dict_re_docs_final['doc'+str(k)]=[x for _, x in sorted(zip(dict_re_docs['tfidf'+str(k)],dict_re_docs['doc'+str(k)]),reverse=True)]
       if len(dict_re_docs_final['doc'+str(k)])>=numbermultiKeywords:
           dict_re_docs_final['doc'+str(k)]= dict_re_docs_final['doc'+str(k)][:numbermultiKeywords]

    dict_uni_re_final=dict(dict_uni_docs_final)

    for key,value in dict_uni_docs_final.items():
        for o in dict_re_docs_final[key]:
           dict_uni_re_final[key].append(o)



    return dict_uni_re_final,dict_re_docs

nuniKey=5
nmultiKey=15
print('start finding Keywords')
Explicit_Keywords,re_in_docs = findExplicit_Keywords(corpus,retfidf_okay,unitfidf_okay,nuniKey,nmultiKey)
print('finished finding Keywords')

with open('Explicit_KeywordsSCP_corpus2mw.txt', 'w') as f:
    for key,value in Explicit_Keywords.items():
        f.write(key)
        f.write(':')
        f.write(' ')
        for v in value:
            f.write(v)
            f.write(';')
            f.write(' ')
        f.write('\n')

def cov(a,b,lencorp):
  i=0
  j=0
  somma=0
  while (j<len(b[::2]) and i<len(a[::2])):
    while (j<len(b[::2]) and a[::2][i]>=b[::2][j]):
      if (a[::2][i]==b[::2][j]):
        somma+=a[2*i+1]*b[2*j+1]
        j=j+1
      else:
        j=j+1

    i=i+1
  return 1000000*somma/(lencorp-1)

def distances(l1,l2,lenre1,lenre2):
    if l1[0]+lenre1<l2[-1]:
      dsmax1=l2[-1]-l1[0]-lenre1
    else:
      dsmax1=l1[0]-l2[-1]-lenre2

    if l2[0]+lenre2<l1[-1]:
      dsmax2=l1[-1]-l2[0]-lenre2
    else:
      dsmax2=l2[0]-l1[-1]-lenre1
    dsmax=max(dsmax1,dsmax2)
    if dsmax==0:
        return 1,0
    dsmin=1000
    for i in range(len(l1)):
        for j in range(len(l2)):
            if l2[j]>l1[i]+lenre1:
              aux=l2[j]-l1[i]-lenre1
            else:
              aux=l1[i]-l2[j]-lenre2
              if aux<dsmin:
                dismin=aux
    return dsmax,dsmin




def ip(a,b,lenre1,lenre2):
  i=0
  j=0
  somma=0
  count=0
  while (j<len(b[::2]) and i<len(a[::2])):
    while (j<len(b[::2]) and a[::2][i]>=b[::2][j]):
      if (a[::2][i]==b[::2][j]):
        dstmax,dstmin=distances(a[2*i+1],b[2*j+1],lenre1,lenre2)
        somma+=dstmin/dstmax
        count+=1
        j=j+1
      else:
        j=j+1

    i=i+1

  return max(0,1-somma/count)

def create_covdictmod(di1,lencorp):
    di2 = dict(di1)
    di3 = {}
    dicorr={}
    count=0
    for key1, value1 in di1.items():
        d = {}
        #d["docs"] = value1[::2]
        for key2, value2 in di2.items():
            # print(value1)
            cv = cov(value1, value2,lencorp)
            # if cv!=0:
            #  ipvalue=ip(dpos[key1],dpos[key2],len(str_to_list(key1)),len(str_to_list(key2)))
            # else:
             #ipvalue=0
            if abs(cv)>0:
               d[key2]=cv#*np.sqrt(ipvalue)
        count+=1
        print('finished key'+str(count))
        di3[key1] = d
        di2.pop(key1)
    print('finished covariance')
    for key3,value3 in di3.items():
        d={}
        covkey3=value3[key3]
        for key4,value4 in value3.items():
            corr=value4/(np.sqrt(covkey3)*np.sqrt(di3[key4][key4]))
            if abs(corr)>0.0000000001:
               d[key4]=corr

        dicorr[key3]=d

    return dicorr

print('start creating dict corr just re')
dict_corr_re=create_covdictmod(reprobs_okay,len(corpus))
print('finished creating dict corr just re')

with open('dict_correlations.txt','w') as file:
    for key,value in dict_corr_re.items():
        file.write(key)
        file.write(": ")
        for key2,value2 in value.items():
            file.write(key2)
            file.write("= ")
            file.write(str(value2))
            file.write("; ")
        file.write("\n")
        file.write("\n")
        file.write("\n")


def findImplicit_Keywords(corpus,dict_cov_re,Explicit_Keywords,re_in_doc,numberImplicitKeywords,firstexplmultikeyword,numkeyscore):
    scores = {}
    for k in range(len(corpus)):
       scores['doc'+str(k)]=[]
       scores['scores'+str(k)]=[]


    for doc in range(len(corpus)):
        for doc1 in range(len(corpus)):
            if (doc != doc1):
                for re in re_in_doc['doc'+str(doc1)]:
                    if (re not in scores['doc'+str(doc)]) and (re not in re_in_doc['doc'+str(doc)]):
                        somma = 0
                        for j in range(min(numkeyscore,len(Explicit_Keywords["doc" + str(doc)][firstexplmultikeyword:]))):
                            if Explicit_Keywords["doc" + str(doc)][j + firstexplmultikeyword] in dict_cov_re[re].keys():
                                somma += dict_cov_re[re][Explicit_Keywords["doc" + str(doc)][j+firstexplmultikeyword]]
                            elif re in dict_cov_re[Explicit_Keywords["doc" + str(doc)][j + firstexplmultikeyword]].keys():
                                somma += dict_cov_re[Explicit_Keywords["doc" + str(doc)][j + firstexplmultikeyword]][re]
                        scores['doc'+str(doc)].append(re)
                        scores['scores'+str(doc)].append(somma / numkeyscore)

        print('finished scores doc'+str(doc))
    dict_implkey_re_final={}
    for k in range(len(corpus)):
       dict_implkey_re_final['doc'+str(k)]=[x for _, x in sorted(zip(scores['scores'+str(k)],scores['doc'+str(k)]),reverse=True)]
       if len(dict_implkey_re_final['doc'+str(k)])>=numberImplicitKeywords:
           dict_implkey_re_final['doc'+str(k)]= dict_implkey_re_final['doc'+str(k)][:numberImplicitKeywords]
       print('finished implict doc'+str(k))



    return dict_implkey_re_final

print('start finding implicit keywords')
Implicit_keywords=findImplicit_Keywords(corpus,dict_corr_re,Explicit_Keywords,re_in_docs,5,5,10)
print('finished finding implicit keywords')


with open('Implicit_KeywordsSCP_corpus2mw.txt', 'w') as f:
    for key,value in Implicit_keywords.items():
        f.write(key)
        f.write(':')
        f.write(' ')
        for v in value:
            f.write(v)
            f.write(';')
            f.write(' ')
        f.write('\n')

# with open('Explicit_KeywordsSCP_corpus2mw.txt', 'w') as f:
#     for key,value in Explicit_Keywords.items():
#         f.write(key)
#         f.write(':')
#         f.write(' ')
#         for v in value:
#             f.write(v)
#             f.write(';')
#             f.write(' ')
#         f.write('\n')



