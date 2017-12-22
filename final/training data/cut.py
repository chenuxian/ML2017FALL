import jieba

jieba.set_dictionary('dict.txt.big')
	
out_f = open("word_vector","w")
with open('1_train.txt', 'r') as f:  
	for line in f:
		word_list = list(jieba.cut(line.strip()))
		out_f.write("%s\n"%" ".join(word_list))
		
with open('2_train.txt', 'r') as f:  
	for line in f:
		word_list = list(jieba.cut(line.strip()))
		out_f.write("%s\n"%" ".join(word_list))
		
with open('3_train.txt', 'r') as f:  
	for line in f:
		word_list = list(jieba.cut(line.strip()))
		out_f.write("%s\n"%" ".join(word_list))
		
with open('4_train.txt', 'r') as f:  
	for line in f:
		word_list = list(jieba.cut(line.strip()))
		out_f.write("%s\n"%" ".join(word_list))
		
with open('5_train.txt', 'r') as f:  
	for line in f:
		word_list = list(jieba.cut(line.strip()))
		out_f.write("%s\n"%" ".join(word_list))