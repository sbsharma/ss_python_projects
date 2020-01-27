import pandas
import os
import nltk

df = pandas.read_excel('test_category.xlsx',sheetname='Nov 2017')
catFileMap = {}
c = 97
d = 97

for i in df.index:
    print(df['Category 1'][i])
    if len(str(df['Category 1'][i]))>0:
        cat_temp = df['Category 1'][i]
    else:
        cat_temp = "NOCAT"
    if cat_temp not in catFileMap:
        temp = chr(c) + chr(d)
        catFileMap[cat_temp] = "c" + temp
        d = d + 1
        if d > 122:
            c = c + 1
            d = 97


print(catFileMap.values())
print(catFileMap.keys())


# To Create Corpus Files
os.mkdir("Categorization_Word_token")
path = './Categorization_Word_token'
os.chdir(path)
cats = open("cats.txt", "wt+", encoding='utf-8')

catFileNames = {}
for i in df.index:
    if len(str(df['Category 1'][i]))>0:
        cat_temp = df['Category 1'][i]
    else:
        cat_temp = "NOCAT"
    val = catFileMap.get(cat_temp)
    if val not in catFileNames:
        catFileNames[val] = str(1).zfill(4)
    else:
        temp_int = int(catFileNames.get(val))
        temp_int = temp_int + 1
        catFileNames[val] = str(temp_int).zfill(4)

    content_text = nltk.word_tokenize("%s"% df['Description'][i])
    content_text = nltk.pos_tag(content_text)
    content_text = ' '.join([nltk.tag.tuple2str(tup) for tup in content_text])
    temp_file = val + catFileNames.get(val)
    f = open(temp_file, 'w', encoding= 'utf-8')
    f.write(content_text)
    f.close()
    cats.write(temp_file + ' ' + str(cat_temp).replace(' ', '_') + '\n')
