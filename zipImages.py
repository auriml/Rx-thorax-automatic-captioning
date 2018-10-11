import zipfile
import glob, os
import pandas as pd

df = pd.read_csv('SJ_chest_x_ray_images_labels_160K.csv', header = 0)


os.chdir("../")
# open the zip file for writing, and write stuff to it
for zipName in list(range(0,50)):
    zipName = str(zipName)
    print(zipName)
    file = zipfile.ZipFile(zipName + ".zip", "w")
    li = df[df.ImageDir == int(zipName)].ImageID
    for imageID in li:
        name = 'SJ/image_dir_processed/'+ imageID
        print(name)
        file.write(name, os.path.basename(name), zipfile.ZIP_DEFLATED)
    file.close()  
    print('Zip done!')  

    # open the file again, to see what's in it

    file = zipfile.ZipFile(zipName + ".zip", "r")
    for info in file.infolist():
        print (info.filename, info.date_time, info.file_size, info.compress_size)

    print("starting to upload zip to gdrive..")
    from subprocess import call
    call(["gdrive","upload",  zipName + ".zip"]) #gdrive upload 0.zip
    print ("Upload done!")

    call(["rm",  zipName + ".zip"]) #rm big zip
    print ("Removed zip!")