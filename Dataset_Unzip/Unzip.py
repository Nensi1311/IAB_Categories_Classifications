#import required modules
import zipfile

#First open the file in read mode
unzip_files = zipfile.ZipFile('Final_IAB.zip','r')

#Now we will unzip and put those files in a directory called extracted_dir
unzip_files.extractall("Final_IAB")
