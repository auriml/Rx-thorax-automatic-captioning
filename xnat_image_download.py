#!/usr/bin/env python
import os
import argparse
import requests, zipfile, io
import pandas as pd

parser = argparse.ArgumentParser(description='Download Rx images from XNAT')
parser.add_argument('-f', metavar='dataset_asoc_csv', type=str, nargs=1,
                    help='filename (default  delimited by ";" with the following fields: Access Number ; MR ID XNAT ; Subject anonymized ; Experiment UID')
parser.add_argument('-r',  default=False  ,
                    help='If true download all images and rewrite downloaded ones. If false skip downloaded images and only download new ones. Default false')
parser.add_argument('-d', metavar='dir_to_save_images', type=str, nargs=1,
                    help='dirname to save images (default  ../SJ/image_dir)')
parser.add_argument('u', metavar='username', type=str, nargs=1,
                    help='XNAT account username')
parser.add_argument('p', metavar='password', type=str, nargs=1,
                    help='XNAT account password')

args = parser.parse_args()
dataset_asoc = args.f[0] if args.f  else  './dataset_asoc.csv'
j_username = args.u[0] if args.u  else  ''
j_password = args.p[0] if args.p  else  ''
image_dir = args.d[0] if args.d  else  '../SJ/image_dir'
replace = args.r



root_url = "https://ceib.cipf.es/xnat"
url = "https://ceib.cipf.es/xnat/j_spring_security_check"
login="Login"

ses = requests.session()
res = ses.get("https://ceib.cipf.es/xnat", verify=False)
sessionId = ses.cookies['JSESSIONID']


payload = {'login': 'Login', 'JSESSIONID' : sessionId, 'SESSION_EXPIRATION_TIMEOUT': False }
r = ses.post(url, data=payload,  auth= (j_username, j_password),verify=False)

if ( os.path.isdir(image_dir) == False) :
    os.mkdir(image_dir)

imagesDF = pd.read_csv(dataset_asoc, sep = ';', )
for row in  imagesDF.iterrows():
    row = row[1]
    exp_id = row[' MR ID XNAT ']
    exist = os.path.exists(image_dir + "/" + str(exp_id))
    exist_1 = (not replace and not os.path.exists(image_dir + "/" + str(exp_id)))
    if replace or (not replace and not os.path.exists(image_dir + "/" + str(exp_id))) :
        r = ses.get(root_url + "/data/archive/projects/padchest_ds17/subjects/"+ row[' Subject anonymized '] +"/experiments/"+ str(exp_id) +"/scans/ALL/files?format=zip", verify =False)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(image_dir)
        print("downloaded: " + str(exp_id))
    else:
        print("skipped: " + str(exp_id))