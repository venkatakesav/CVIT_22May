
import os
from shutil import copy2
import time
import random

random.seed = int(time.time())

path = 'D:\\pattern dataset\\xml'
test_cases_path = 'D:\\pattern dataset\\IAMdataset'
images_path = 'D:\\pattern dataset\\forms'
padding = 3
dir_iterator = 1
string_to_search = "<form created"
# writer_ids = []
filename_map = {}

for filename in os.listdir(path):
     with open(os.path.join(path, filename), 'r') as read_obj:
         for line in read_obj:
             if string_to_search in line:
                 leftline,rightline = line.split('writer-id="')
                 answer,junk = rightline.split('">')
                 if (answer != '000'):
                     filename,extension = filename.split('.')
                     # item = [answer,filename]
                     # writer_ids.append(item)
                     if answer in filename_map.keys():
                         filename_map[answer].append(filename)
                     else:
                         filename_map[answer] = [filename]
                 break
                 
                 
# writer_ids.sort()
# for item in writer_ids:
#     if item[0] == '000':
#         writer_ids.remove(item)

# for idnum,name in writer_ids:
#     if idnum in filename_map.keys():
#         filename_map[idnum].append(name)
    # else:
    #     filename_map[idnum] = [name]
    
    
writersWith23forms = dict(filter(lambda elem: len(elem[1]) >= 2,filename_map.items()))
writersWith3Forms = dict(filter(lambda elem: len(elem[1]) >= 3,filename_map.items()))
keyslist2 = list(writersWith23forms.keys())
keyslist3 = list(writersWith3Forms.keys())
# final_itr = len(keyslist3)//3

testCasesNum = 40
answerlist = []

for i in range(testCasesNum):
    # if (i+1 == final_itr):
    #     break

    ## creating a test case folder
    directory = format(dir_iterator, '03d')
    dir_iterator+=1
    full_path = os.path.join(test_cases_path,directory)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
        
    testCaseIndeces = []
    while len(testCaseIndeces) != 3:
        r = random.choice(keyslist3)
        if r not in testCaseIndeces: testCaseIndeces.append(r)
    for j in range (1,4):
        newpath = os.path.join(full_path,str(j))
        if not os.path.exists(os.path.join(full_path,str(j))):
            os.mkdir(os.path.join(full_path,str(j)))
        image_name1 = writersWith3Forms[testCaseIndeces[j-1]][0] + '.png'
        image_name2 = writersWith3Forms[testCaseIndeces[j-1]][1] + '.png'
        copy2(os.path.join(images_path,image_name1), os.path.join(newpath,'1.png'))
        copy2(os.path.join(images_path,image_name2), os.path.join(newpath,'2.png'))
    
    rng = random.randrange(0,3)
    answerlist.append(rng+1)
    test_image = writersWith3Forms[testCaseIndeces[rng]][2] + '.png'
    copy2(os.path.join(images_path,test_image),os.path.join(full_path,'test.png'))

# file_writer = open(path+"\\writers.txt","w")
# for item in writer_ids:
#         file_writer.write(item[0] + " " + item[1] + "\n")
if os.path.exists(os.path.join(test_cases_path,'asnwers.txt')):
    os.remove(os.path.join(test_cases_path,'asnwers.txt'))
file_writer = open(os.path.join(test_cases_path,'asnwers.txt'),'w')
for item in answerlist:
    file_writer.write(str(item) + '\n')
file_writer.close()    
    