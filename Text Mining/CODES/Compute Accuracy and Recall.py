scores=[]
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
def list_to_str(word):
    res = ""
    for a in range(len(word) - 1):
        res += word[a] + " "
    res += word[-1]
    return res
with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/RE_SCP_random_100.txt' , 'r', errors='ignore') as file:
    for line in file:
        #print(str_to_list(line))
        if str_to_list(line)[-1][0]=='1':
          scores.append(1)
        else:
          scores.append(0)

accuracy_SCP=sum(scores)/len(scores)

rerecall=[]
with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/recall_list_corpus2mw.txt','r') as file:
    for line in file:
        auxlist=[]
        for word in line.split():
            auxlist.append(word)
        rerecall.append(list_to_str(auxlist))

retotal=[]
with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/RE_SCP_corpus2mw_unfiltered.txt','r') as file:
    for line in file:
        auxlist=[]
        for word in line.split():
            auxlist.append(word)
        retotal.append(list_to_str(auxlist))

recall_scores=[]
for i in rerecall:
    if i in retotal:
        recall_scores.append(1)
        print(i)
    else:
        recall_scores.append(0)

recall_SCP=sum(recall_scores)/len(recall_scores)






scores=[]

with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/RE_Dice_random_100.txt' , 'r', errors='ignore') as file:
    for line in file:
        #print(str_to_list(line))
        if str_to_list(line)[-1][0]=='1':
          scores.append(1)
        else:
          scores.append(0)

accuracy_Dice=sum(scores)/len(scores)

retotal = []
with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/RE_Dice_corpus2mw_unfiltered.txt',
          'r') as file:
    for line in file:
        auxlist = []
        for word in line.split():
            auxlist.append(word)
        retotal.append(list_to_str(auxlist))

recall_scores = []
for i in rerecall:
    if i in retotal:
        recall_scores.append(1)
        print(i)
    else:
        recall_scores.append(0)

recall_Dice = sum(recall_scores) / len(recall_scores)

scores = []

with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/RE_MI_random_100.txt' , 'r', errors='ignore') as file:
    for line in file:
        #print(str_to_list(line))
        if str_to_list(line)[-1][0]=='1':
          scores.append(1)
        else:
          scores.append(0)

accuracy_MI= sum(scores) / len(scores)

retotal = []
with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/RE_MI_corpus2mw_unfiltered.txt',
          'r') as file:
    for line in file:
        auxlist = []
        for word in line.split():
            auxlist.append(word)
        retotal.append(list_to_str(auxlist))

recall_scores = []
for i in rerecall:
    if i in retotal:
        print(i)
        recall_scores.append(1)
    else:
        recall_scores.append(0)

recall_MI = sum(recall_scores) / len(recall_scores)

scores = []


rerecall=[]
with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/recall_list_corpus2mw.txt','r') as file:
    for line in file:
        auxlist=[]
        for word in line.split():
            auxlist.append(word)
        rerecall.append(list_to_str(auxlist))

with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/RE_SCP_random_100bis.txt' , 'r', errors='ignore') as file:
    for line in file:
        #print(str_to_list(line))
        if str_to_list(line)[-1][0]=='1':
          scores.append(1)
        else:
          scores.append(0)

accuracy_SCPbis = sum(scores) / len(scores)

scores = []

with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/RE_Dice_random_100bis.txt' , 'r', errors='ignore') as file:
    for line in file:
        #print(str_to_list(line))
        if str_to_list(line)[-1][0]=='1':
          scores.append(1)
        else:
          scores.append(0)

accuracy_Dicebis = sum(scores) / len(scores)

scores = []
with open('c:/Users/danie/Desktop/POLI/Erasmus Nova/Data Analytics and Mining/PART II/Project/RE_MI_random_100bis.txt', 'r', errors='ignore') as file:
    for line in file:
        if str_to_list(line)[-1][0]=='1':
          scores.append(1)
        else:
          scores.append(0)

accuracy_MIbis = sum(scores) / len(scores)

