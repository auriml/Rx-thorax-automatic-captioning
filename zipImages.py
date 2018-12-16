import zipfile
import glob, os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Util methods')

parser.add_argument('-fr', metavar='Start_From_Zip_Number', type=int, nargs=1,
                    help='start generating and uploading zip file with the given number')
parser.add_argument('-to', metavar='To_Folder_Number', type=int, nargs=1,
                    help='start generating and uploading zip file with up to given number')

args = parser.parse_args()
fro = args.fr[0] if args.fr  else  None
to = args.to[0] if args.to  else  None

os.chdir("../")
df = pd.read_csv('Rx-thorax-automatic-captioning/SJ_chest_x_ray_images_labels_160K.csv', header = 0)



# open the zip file for writing, and write stuff to it
for zipName in list(range(fro,to)):
    zipName = str(zipName)
    print(zipName)
    file = zipfile.ZipFile('SJ/tmp/' + zipName + ".zip", "w")
    li = df[df.ImageDir == int(zipName)].ImageID
    for imageID in li:
        name = 'SJ/image_dir_processed/'+ imageID
        print(name)
        file.write(name, os.path.basename(name), zipfile.ZIP_DEFLATED)
    file.close()  
    print('Zip done!')  

    # open the file again, to see what's in it

    #file = zipfile.ZipFile('SJ/' +zipName + ".zip", "r")
    #for info in file.infolist():
    #    print (info.filename, info.date_time, info.file_size, info.compress_size)

    print("starting to upload zip to gdrive..")
    from subprocess import call
    call(["gdrive","upload",  'SJ/tmp/' + zipName + ".zip"]) #gdrive upload 0.zip
    print ("Upload done!")

    call(["rm",  'SJ/tmp/' +zipName + ".zip"]) #rm big zip
    print ("Removed zip!")