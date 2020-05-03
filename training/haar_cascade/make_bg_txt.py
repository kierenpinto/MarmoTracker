import os
text_file = 'bg.txt'
fp = open(text_file,'w')
directory = 'background_images/neg/'
photo_list = os.listdir('./' + directory)
for file in photo_list:
    file_name = directory + file + '\n'
    fp.write(file_name)

