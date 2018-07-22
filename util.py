#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
from skimage import io, exposure, transform, img_as_float
import pydicom
import re


import os

import requests, zipfile
import pandas as pd
import matplotlib.pyplot as plt
import re
import yaml
from anytree import AnyNode
from anytree.exporter import DictExporter
from anytree.importer import DictImporter
from pprint import pprint  # just for nice printing
from anytree import RenderTree , search # just for nice printing
import remotedebugger as rd


parser = argparse.ArgumentParser(description='Util methods')
rd.attachDebugger(parser)
parser.add_argument('-s', metavar='MR_ID_XNAT', type=str, nargs=1,
                    help='display DICOM info for all images of a study identified with MR ID XNAT')
parser.add_argument('-ps', metavar='MR_ID_XNAT', type=str, nargs=1,
                    help='display DICOM info for all images of all studies available for a patient that has a study identified with MR ID XNAT')
parser.add_argument('-p', metavar='PatientID', type=str, nargs=1,
                    help='display DICOM info for all images of all studies available for a patient identified with patientID')

parser.add_argument('-m', metavar='DICOM field modality',
                    help='summarize all values in dataset for the DICOM field modality')
parser.add_argument('-f',  metavar='filename', type=str, nargs='?', default= True,
                    help='save in filename all info for all studies. Default filename: /all_info_studies.csv  ')
parser.add_argument('-d',  metavar='filename', type=str, nargs='?', default= True,
                    help='describe all info for all studies. Default filename: /all_info_studies.csv  ')
parser.add_argument('-dc',  metavar='categorical_field', type=str,
                    help='summarize DICOM field categories for all studies. Default filename: /all_info_studies.csv. Possible categorical fields are '
                         'all,Modality,SeriesDescription,ProtocolName, BodyPartExamined,ViewPosition, CodeMeaning,PhotometricInterpretation, Manufacturer')
parser.add_argument('-dn',  metavar='numerical_field', type=str,
                    help='summarize DICOM numerical field  for all studies. Default filename: /all_info_studies.csv. Possible numerical fields are '
                         'PatientBirth,StudyDate,Rows,Columns,PixelAspectRatio,SpatialResolution, XRayTubeCurrent,ExposureTime, ExposureInuAs,Exposure, RelativeXRayExposure, BitsStored, PixelRepresentation, WindowCenter, WindowWidth')

parser.add_argument('-split',action='store_true',
                    help='split in side and front views based on DICOM info. Default source filename: /all_info_studies.csv.')

parser.add_argument('-e',action='store_true',
                    help='generate list of images to exclude. It excludes non Rx thorax studies or if patient position is not vertical or if image has not associated report. Default source filename: /all_info_studies.csv.')

parser.add_argument('-imgs', metavar='MR_ID_XNAT', type=str, nargs=1,
                    help='preprocess all images of a study identified with MR ID XNAT')

parser.add_argument('-imgm', metavar='n_samples', type=int, nargs=1,
                    help='generate mean X-Ray picture and mean standard deviation picture for n samples')
parser.add_argument('-fst',  metavar='filename', type=str, nargs='?', default= True,
                    help='save in filename all info for all studies by sentence topic. Default filename: /all_info_studies_sent_topics.csv  ')
parser.add_argument('-est',  metavar='filename', type=str, nargs='?', default= True,
                    help='extract and save in filename sentence topics. Default filename: /extract_sent_topics.csv  ')



#parser.add_argument('u', metavar='username', type=str, nargs=1,
                    #help='XNAT account username')
#parser.add_argument('p', metavar='password', type=str, nargs=1,
                    #help='XNAT account password')
all_info_studies_file = '/all_info_studies.csv'
all_info_studies_st_file = 'all_info_studies_sent_topics.csv'
extract_topics_file = 'extract_sent_topics.csv'

args = parser.parse_args()
ID_XNAT = args.s[0] if args.s  else  None
patient_ID_XNAT = args.ps[0] if args.ps  else  None
patient_ID = args.p[0] if args.p  else  None
modality = args.m if args.m  else  None
filename =  all_info_studies_file if args.f is None else None
filename_st =  all_info_studies_st_file if args.fst is None else None
filename_to_describe =  all_info_studies_file if args.d is None else None
categorical_field = args.dc if args.dc  else  None
numerical_field = args.dn if args.dn  else  None
split_images_side_front = args.split if args.split  else  None
exclude = args.e if args.e  else  None
imgs_ID_XNAT = args.imgs[0] if args.imgs  else  None
sample_image = args.imgm[0] if args.imgm  else  None
extract_topics = extract_topics_file if args.est is None else None


#j_username = args.u[0] if args.u  else  ''
#j_password = args.p[0] if args.p  else  ''


currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)

#get dictionary of image paths and image modality (CT, RX, DX) for a given study identified with ID XNAT
def getImagesPath(ID_XNAT ):
    path = root +  '/SJ/image_dir/'
    fieldValues = {}  # create an empty field dictionary of values
    walk_path = path  if not ID_XNAT else path + ID_XNAT
    for dirName, subdirList, fileList in os.walk(walk_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                filename = os.path.join(dirName,filename)
                e = np.fromfile(filename, dtype='>u2')
                try:
                    RefDs = pydicom.read_file(filename)

                    value = RefDs.Modality
                    fieldValues[filename] = value

                except:

                    pass

                print(fieldValues)

    print(fieldValues)

    return fieldValues



def getDicomInfo(ID_XNAT):
    path = root + '/SJ'
    walk_path = path + '/image_dir/' + ID_XNAT
    images = {}
    print(str(ID_XNAT))
    for dirName, subdirList, fileList in os.walk(walk_path ):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                try:
                    RefDs = pydicom.read_file(os.path.join(dirName,filename),force=True)
                    images['/image_dir_processed/' + ID_XNAT + '_' + filename[-10:-4] + '.png'] = RefDs
                except:
                    print('Failed opening DICOM file')
                    pass
    return images

# Get report + images + Dicom info for a given study identified by ID XNAT
# Precondition: Images need to be previously preprocessed and stored in image_dir
# Arguments: The study identified by MR ID XNAT e.g 171456198269648269029527880939880883235
# Return: the list of image dictionaries that belongs to a study. Each dict contains  patient's and image information fields
def getAllInfo(ID_XNAT, dataset_asoc_DF = None, report_DF = None):
    study_info = {}
    #access number of report
    dataset_asoc = '/dataset_asoc_abril18.csv'
    path = root + '/Rx-thorax-automatic-captioning' + dataset_asoc
    dataset_asoc_DF = pd.read_csv(path, sep = ';', ) if  dataset_asoc_DF is None else dataset_asoc_DF
    exp_id = dataset_asoc_DF.loc[dataset_asoc_DF['MR ID XNAT'] == ID_XNAT]

    study_info['codigoinforme'] = None
    #access report
    try:
        #access number of report
        dataset_asoc = '/dataset_asoc_abril18.csv'
        path = root + '/Rx-thorax-automatic-captioning' + dataset_asoc
        dataset_asoc_DF = pd.read_csv(path, sep = ';', ) if  dataset_asoc_DF is None else dataset_asoc_DF
        exp_id = dataset_asoc_DF.loc[dataset_asoc_DF['MR ID XNAT'] == ID_XNAT]
        exp_id = int(exp_id.iloc[0]['Access Number'])
        study_info['codigoinforme'] = int(str(exp_id)[-7:])
        #access report
        filename = '/report_sentences_preprocessed.csv'
        path = root + '/Rx-thorax-automatic-captioning' + filename
        report_DF = pd.read_csv(path, sep = ',', encoding='ISO-8859-1') if  report_DF is None else report_DF
        text = report_DF.loc[report_DF['codigoinforme'] == study_info['codigoinforme']]

        study_info['report'] = text.iloc[0]['v_preprocessed']
    except:
        study_info['report'] = None
        print("MISSING REPORT: " +str(ID_XNAT) + ' ' + str(study_info['codigoinforme']))
        pass

    print(study_info)

    #access study's images
    images =[]
    dicoms = getDicomInfo(ID_XNAT)
    pattern = re.compile('[\W_]+')
    for key, value in dicoms.items():
        print(value)
        images_info = {}
        images_info["ImagePath"] = key #TODO: image path should be the preprocessed image path not the dcm path
        images_info["StudyID"] = ID_XNAT #A study has 1 to Many images
        images_info["PatientID"] = value.PatientName  if 'PatientName' in value else None #codified patient's name //TODO: use it to ensure that the same patient is not both in the training and testing sets to avoid overfitting
        images_info["PatientBirth"] = str(value.PatientBirthDate)[:4] if 'PatientBirthDate' in value else None
        images_info["PatientSex"] = value.PatientSex if 'PatientSex' in value else None
        images_info['ReportID'] = study_info['codigoinforme']
        images_info['Report'] = study_info['report']
        images_info['StudyDate'] = value.StudyDate if 'StudyDate' in value else None
        images_info['Modality'] = pattern.sub('', value.Modality) if 'Modality' in value else None
        images_info['SeriesDescription'] = pattern.sub('', value.SeriesDescription) if 'SeriesDescription' in value else None
        images_info['ProtocolName'] = pattern.sub('', value.ProtocolName) if 'ProtocolName' in value else None
        images_info['CodeMeaning'] = pattern.sub('', value.ProcedureCodeSequence[0].CodeMeaning) if 'ProcedureCodeSequence' in value and 'CodeMeaning' in value.ProcedureCodeSequence[0] else None
        images_info['Manufacturer'] = pattern.sub('', value.Manufacturer) if 'Manufacturer' in value else None

        images_info['ViewPosition'] = pattern.sub('', value.ViewPosition) if 'ViewPosition' in value else None
        images_info['BodyPartExamined'] = pattern.sub('', value.BodyPartExamined) if 'BodyPartExamined' in value else None #After reviewing this field it is innacurate and inconsistent

        images_info['Rows'] = value.Rows if 'Rows' in value else None
        images_info['Columns'] = value.Columns if 'Columns' in value else None
        images_info['PixelAspectRatio'] = value.PixelAspectRatio if 'PixelAspectRatio' in value else None
        images_info['SpatialResolution'] = value.SpatialResolution if 'SpatialResolution' in value else None
        images_info['PhotometricInterpretation'] = pattern.sub('', value.PhotometricInterpretation)  if 'PhotometricInterpretation' in value else None
        images_info['BitsStored'] = value.BitsStored  if 'BitsStored' in value else None
        images_info['PixelRepresentation'] = value.PixelRepresentation  if 'PixelRepresentation' in value else None
        images_info['WindowCenter'] = value.WindowCenter  if 'WindowCenter' in value else None
        images_info['WindowWidth'] = value.WindowWidth  if 'WindowWidth' in value else None

        images_info['RelativeXRayExposure'] = value.RelativeXRayExposure if 'RelativeXRayExposure' in value else None
        images_info['ExposureTime'] = value.ExposureTime if 'ExposureTime' in value else None #Expressed in sec
        images_info['XRayTubeCurrent'] = value.XRayTubeCurrent if 'XRayTubeCurrent' in value else None
        images_info['ExposureInuAs'] = value.ExposureInuAs if 'ExposureInuAs' in value else None
        images_info['Exposure'] = value.Exposure if 'Exposure' in value else None

        images.append(images_info)

    return images

def saveAllStudyInfoFullDataset(save_file = all_info_studies_file, dataset_asoc_file = '/dataset_asoc_10042018.csv'):
    dataset_asoc = '/dataset_asoc_10042018.csv' if not dataset_asoc_file else dataset_asoc_file
    path = root + '/Rx-thorax-automatic-captioning' + dataset_asoc
    dataset_asoc_DF = pd.read_csv(path, sep = ',', )

    reports = '/report_sentences_preprocessed.csv'
    path = root + '/Rx-thorax-automatic-captioning' + reports
    report_DF = pd.read_csv(path, sep = ',', encoding='ISO-8859-1')


    all_studies_DF = None
    empty = False
    path = root + '/Rx-thorax-automatic-captioning' + save_file
    if os.path.exists(path):
        all_studies_DF = pd.read_csv(path, sep = ';' , header = 0)
    else:
        empty = True
    f = open(path, 'a')


    for index, r in dataset_asoc_DF.iterrows():
        studyID = r['MR ID XNAT']
        if  all_studies_DF is None or not all_studies_DF['StudyID'].str.contains(studyID).any():
            imagesInfo = getAllInfo(studyID,dataset_asoc_DF, report_DF)
            for img in imagesInfo:
                if empty:
                    f.write(";".join(img.keys()) + '\n')
                    empty = False
                f.writelines(";".join(str(e) for e in img.values()) + '\n')

    f.close()





def summarizeAllStudiesDicomModality():
    path = root +  '/SJ/image_dir/'
    fieldValues = {}  # create an empty field dictionary of values
    samples = {}
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                filename = os.path.join(dirName,filename)
                e = np.fromfile(filename, dtype='>u2')
                try:
                    RefDs = pydicom.read_file(filename)

                    value = RefDs.Modality
                    if value in fieldValues.keys():
                        fieldValues[value] +=  1
                    else:
                        fieldValues[value] = 1
                        samples[value] = filename
                except:

                    pass

                print(fieldValues)
                print(samples)
    print(fieldValues)
    print(samples)
    return fieldValues, samples

def splitImagesFrontalSide(file = all_info_studies_file):
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + file, sep = ';' , header = 0)
    #exclude non-evaluable images
    if os.path.exists('Excluded_images.csv'):
        excluded = pd.read_csv('Excluded_images.csv', sep = ',' , header = 0)
    else:
        excluded = generateExcludedImageList()
    idx = all_studies_DF[all_studies_DF['ImagePath'].isin(excluded['ImagePath'])].index.values
    print("Number of excluded images: " + str(len(idx)))
    all_studies_DF = all_studies_DF.drop(idx)

    #Side view images
    # where StudyDescription is in lat array (a manual selection from values of StudyDescription,_
    # please run summarizeAllStudiesByCategory when new images are added to dataset to identify new values)
    lat = ["Lateral","Lateralizq", "LatVertical", "LatHorizontal", "Decblatizq"]
    side_images = all_studies_DF[all_studies_DF['SeriesDescription'].isin(lat)]

    # where ViewPosition is in lat array
    lat = ['LATERAL','LL','LLD','RL'] #LL and RL means right lateral view (all other are left lateral views)
    side_images = side_images.append(all_studies_DF[all_studies_DF['ViewPosition'].isin(lat)])
    side_images = side_images.groupby('ImagePath').first()

    #TODO: add those where image_dir path contain "lat" ? not a good method, is not exhaustive

    #where Code Meaning is in
    lat = ['RXTORAXPAYLAT','RXTORAXPAYLATPED'] #Those are the studies with both frontal and side views, so to not add them but compare with prior figure
    both = all_studies_DF[all_studies_DF['CodeMeaning'].isin(lat)]
    uniqueBoth = both.groupby('ImagePath').first()
    studies_to_review = uniqueBoth[~uniqueBoth['StudyID'].isin(side_images['StudyID'])]
    #side_images = side_images.append(all_studies_DF[all_studies_DF['CodeMeaning'].isin(lat)])
    #side_images = side_images.groupby('ImagePath').first()


    #Front view images where StudyDescription is in front array (a manual selection from values of StudyDescription,
    # please run summarizeAllStudiesByCategory when new images are added to dataset to identify new values)
    front = ["Trax","Tórax","PA", "PAhoriz","APhorizontal","PAvertical", "pulmon", "AP","torax","APhoriz", "APvertical",
    "Lordtica", "APHorizontal", "PAHorizontal","Pediatra3aos", "Pediatría3años","APVertical", "Pedit3aos",  "Pediát3años", "W033TraxPA"]
    front_images = all_studies_DF[all_studies_DF['SeriesDescription'].isin(front)] #Evaluated and Not reliable

    front_images = all_studies_DF[~all_studies_DF['ImagePath'].isin(side_images.index.values)]
    front_images = front_images[~front_images['StudyID'].isin(studies_to_review['StudyID'])]

    side_images.to_csv('position_side_images.csv')
    side_images[side_images['ViewPosition'].isin(['LLD','RL'])].to_csv('position_side_right_images.csv')
    front_images.to_csv('position_front_images.csv')
    studies_to_review.to_csv('position_toreview_images.csv')

    return {'side' : side_images, 'front': front_images }

def summarizeAllStudiesByCategoricalField (file = all_info_studies_file, categorical_field = None):
    #Return dataframe where each row is one class and  values contains the number of ocurrences and one example
    #Possible categorical fields are: 'Modality;SeriesDescription;ProtocolName;ViewPosition;PhotometricInterpretation'
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + file, sep = ';' , header = 0)
    n = all_studies_DF.groupby(categorical_field).first()
    c = all_studies_DF[categorical_field].value_counts()
    n = n.join(c)

    return n
def summarizeAllStudiesByNumericalField (file = all_info_studies_file, numerical_field = None):
    #Return dictionary where each key is one class and each value is a tuple with number of ocurrences and one example
    dict = {}
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + file, sep = ';' , header = 0)
    n = all_studies_DF[numerical_field]
    n = pd.to_numeric(n, errors='coerce')


    return n.describe()
def summarizeAllStudies(file = all_info_studies_file):
    summary = {}
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + file, sep = ';' , header = 0)

    #exclude non-evaluable images
    if os.path.exists('Excluded_images.csv'):
        excluded = pd.read_csv('Excluded_images.csv', sep = ',' , header = 0)
    else:
        excluded = generateExcludedImageList()
    idx = all_studies_DF[all_studies_DF['ImagePath'].isin(excluded['ImagePath'])].index.values
    print("Number of excluded images: " + str(len(idx)))
    all_studies_DF = all_studies_DF.drop(idx)

    #Number of different images
    n = all_studies_DF['ImagePath'].count()
    print("Number of different images: " + str(n))
    #Number of different patients
    np = all_studies_DF.groupby(['PatientID']).count()
    print("Number of different patients: " + str(len(np)))
    print ("Distribution of # of images by patient")
    print ( np['StudyID'].describe())
    #ax = serie.hist()
    #fig = ax.get_figure()
    #fig.savefig('PatientNumberDistribution.pdf')
    #import matplotlib.pyplot as plt
    #s.hist()
    #plt.savefig('path/to/figure.pdf')  # saves the current figure

    #Number of studies per patient (mean, min, max)
    ns = all_studies_DF.groupby(['StudyID']).count()
    print("Number of different studies: " + str(len(ns)))
    #Number of images per study (mean, min, max)
    print("Distribution of # of images per study (mean, min, max): ")
    ns = all_studies_DF.groupby(['StudyID'])['ImagePath'].nunique()
    print(ns.describe())
    ns = all_studies_DF.groupby(['PatientID'])['StudyID'].nunique()
    print("Distribution of # of studies per patient (mean, min, max): ")
    print(ns.describe())
    patients_with_multiple_studies= ns[ns > 1]
    patients_with_multiple_studies.hist()
    plt.title("Histogram of Patients with Multiple Studies (n = " + str(patients_with_multiple_studies.count()) + ")" )
    plt.xlabel("Number of Studies")
    plt.ylabel("Number of Patients")
    plt.savefig('graphs/StudiesPerPatientHistogram.png')
    plt.gcf().clear()
    patients_with_multiple_studies.to_csv('Patients_with_multiple_studies.csv')


    #Birth year (mean, min, max), (distribution by year)
    print ('Birth Year Distribution: ')
    years = all_studies_DF['PatientBirth']
    years = pd.to_numeric(years, errors='coerce')
    print (years.describe())
    years.hist()
    plt.title("Birth Year Histogram (n = " + str(n) + ")" )
    plt.xlabel("Patient's Birth Year")
    plt.ylabel("Images")
    plt.savefig('graphs/BirthYearHistogram.png')  # saves the current figure
    plt.gcf().clear()

    #Age year  (mean, min, max), (distribution by year)
    print ('Age Distribution: ')
    study_dates = all_studies_DF['StudyDate'].apply(lambda x: str(x)[:4])
    ages = pd.to_numeric(study_dates, errors='coerce' ) -  years
    print (ages.describe())
    #ages.hist()
    freq = pd.Series(ages).value_counts().sort_index()
    freq.plot(kind='bar', color='blue', grid=False)
    plt.title("Age Histogram (n = " + str(n) + ")" )
    plt.xlabel("Patient's Age")
    plt.xticks([0,10,20,30,40,50,60,70,80,90,100], [0,10,20,30,40,50,60,70,80,90,100])
    plt.ylabel("Images")
    plt.savefig('graphs/AgeHistogram.png')
    plt.gcf().clear()

    #Sex
    print("Gender Histogram (n = " + str(n) + ")")
    sex = all_studies_DF['PatientSex']
    sex.hist()
    plt.title("Gender Histogram (n = " + str(n) + ")" )
    plt.xlabel("Gender")
    plt.ylabel("Images")
    plt.savefig('graphs/GenderHistogram.png')
    plt.gcf().clear()

    #Sex by age
    print("Age by Gender Histogram (n = " + str(n) + ")")
    ages_sex = all_studies_DF.join(pd.DataFrame({'PatientAge' :ages}))
    ages_sex = ages_sex[ages_sex["PatientSex"].isin(['F','M'])]
    table_ages_sex = pd.crosstab(index=ages_sex["PatientAge"], columns=ages_sex["PatientSex"])
    table_ages_sex.plot(kind='bar', stacked=False, color=['red','blue'], grid=False)

    plt.title("Ages by Gender Histogram (n = " + str(n) + ")" )
    plt.xlabel("Ages")
    plt.xticks([0,10,20,30,40,50,60,70,80,90,100], [0,10,20,30,40,50,60,70,80,90,100])
    plt.ylabel("Images")
    plt.savefig('graphs/AgesByGenderHistogram.png')
    plt.gcf().clear()

    table_ages_sex.plot(kind='box')
    plt.title("Gender Ages Boxplot (n = " + str(n) + ")" )
    plt.xlabel("Gender")
    plt.ylabel("Images")
    plt.savefig('graphs/AgesByGenderBoxplot.png')
    plt.gcf().clear()

    #Study date (distribution by year)
    print('Study Date Distribution')
    study_dates = pd.to_numeric(all_studies_DF['StudyDate'], errors='coerce' ).dropna()
    study_dates = study_dates.apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8]).astype('datetime64[ns]')
    years = pd.DatetimeIndex(study_dates).year.values
    pd.Series(years).hist()
    plt.title("Study Year Histogram (n = " + str(n) + ")" )
    plt.xlabel("Year of Study")
    plt.ylabel("Images")
    plt.savefig('graphs/StudyYearHistogram.png')
    plt.gcf().clear()


    months = pd.DatetimeIndex(study_dates).month.values
    month_names = ['JAN', 'FEB', 'MAR', 'APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    freq = pd.Series(months).value_counts().sort_index()
    freq.plot(kind='bar')
    plt.title("Study Month Histogram (n = " + str(n) + ")" )
    plt.xlabel("Month of Study")
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], month_names)
    plt.ylabel("Images")
    plt.savefig('graphs/StudyMonthHistogram.png')
    plt.gcf().clear()


    #Study Type distribution (portatil ['RXSIMPLETORAXPORTATIL1OMASPROYECCIONES','RXTORAXCONPORTATIL'], pediatric ['TORAXPEDIATRICO,'RXTORAXPAYLATPED']
    print("Study Type Histogram (n = " + str(n) + ")")
    type = all_studies_DF['CodeMeaning']
    #type['TypeCat'] = type[type['CodeMeaning'].isin(['RXSIMPLETORAX1OMASPROYECCIONES','RXTORAXLORDOTICAS'])] #Frontal
    #type['TypeCat'] = type[type['CodeMeaning'].isin(['RXSIMPLETORAXPORTATIL1OMASPROYECCIONES','RXTORAXCONPORTATIL'])] #Portatil
    #type['TypeCat'] = type[type['CodeMeaning'].isin(['RXTORAXDLCONRAYOHORIZONTAL'])]#Lateral Izquierda
    type.hist()
    plt.title("Study Type Histogram  (n = " + str(n) + ")" )
    plt.xlabel("Study Type")
    plt.ylabel("Images")
    plt.savefig('graphs/TypeHistogram.png')
    plt.gcf().clear()

    #Radiation exposure levels by study year
    print("Radiation exposure levels by study year")
    temp = pd.DataFrame()
    temp['Year'] = pd.to_numeric(all_studies_DF['StudyDate'], errors='coerce' )
    temp['Exposure'] = pd.to_numeric(all_studies_DF['Exposure'], errors='coerce', downcast='integer' )
    print(temp['Exposure'].describe())
    temp= temp.dropna()
    temp = temp[temp['Exposure'] > 0]
    temp['Year'] = temp['Year'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8]).astype('datetime64[ns]')
    temp['Year'] = pd.DatetimeIndex(temp['Year']).year.values
    pivot = temp.pivot(columns='Year', values='Exposure')
    pivot.boxplot()
    plt.title("Radiation Exposure by Study Year  (n = " + str(n) + ")" )
    plt.xlabel("Years")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByYear.png')
    plt.gcf().clear()
    pivot.boxplot(showfliers=False)
    plt.title("Radiation Exposure by Study Year  (n = " + str(n) + ")" )
    plt.xlabel("Years")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByYear_NotOutliers.png')
    plt.gcf().clear()


    #Radiation exposure levels by type of Rx (Lateral vs AP)
    position = pd.read_csv('position_side_images.csv')
    position['Position'] = 'side'
    position = pd.concat([position, pd.read_csv('position_front_images.csv')], axis = 0)
    position.Position.ix[position['Position'] != 'side'] = 'front'
    print("Type of position views: frontal vs lateral")
    print(position.Position.describe())
    n_by_position = position.groupby('Position')['ImagePath'].apply(lambda x: x.count())
    n_side = n_by_position.ix['side']
    n_front = n_by_position.ix['front']
    print(n_by_position)
    position['Exposure'] = pd.to_numeric(position['Exposure'], errors='coerce', downcast='integer' )

    pivot = position.pivot(columns='Position', values='Exposure')
    pivot.boxplot()
    plt.title("Radiation Exposure by Position View  (# front = " + str(n_front) + ", # side = "+ str(n_side) )
    plt.xlabel("Position View")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByPositionView.png')
    plt.gcf().clear()

    pivot = position.pivot(columns='Position', values='Exposure')
    pivot.boxplot(showfliers = False)
    plt.title("Radiation Exposure by Position View  (# front = " + str(n_front) + ", # side = "+ str(n_side) )
    plt.xlabel("Position View")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByPositionViewWithoutOutlayer.png')
    plt.gcf().clear()

    position = pd.concat([position, pd.read_csv('position_toreview_images.csv')], axis = 0)
    position.Position.ix[position['Position'].isna()] = 'unk'
    position.Position.hist()
    plt.title("Position View Histogram  (n = " + str(n) + ")" )
    plt.xlabel("Position View")
    plt.ylabel("Images")
    plt.savefig('graphs/PositionViewHistogram.png')
    plt.gcf().clear()

    #Dynamic Range Distribution by type of Rx (Lateral vs AP)






    return summary

def generateMeanXRay(sample):
    #load a sample (n =100 ) of frontal views
    n_sample = sample if sample else 100
    front_studies = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + '/position_front_images.csv',  header = 0)
    front_studies = front_studies.sample(n = n_sample,random_state=3)
    concatenated_img = np.full((400, 400,1), 0) #remove
    for path in front_studies['ImagePath']:
        path = root + '/SJ' + path #change
        img = img_as_float(io.imread(path))
        img = transform.resize(img, (400,400))
        img = np.expand_dims(img, -1)
        img = np.uint8(img * 255)

        concatenated_img = np.concatenate((concatenated_img,img), axis=2)

    #calculate mean
    mean = np.uint8(np.mean(concatenated_img, axis=2))
    img = exposure.equalize_hist(mean)
    io.imsave('graphs/MeanImage.png',img)
    std = np.uint8(np.std(concatenated_img, axis=2))
    img = exposure.equalize_hist(std)
    io.imsave('graphs/StdImage.png',img)

    return


def getAllInfoPatient(patient_ID_XNAT= False , patient_ID=False):
    pd.set_option('display.max_colwidth', -1)
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + all_info_studies_file , sep = ';' , header = 0)
    rows = None
    if patient_ID_XNAT:
        rows = all_studies_DF.loc[all_studies_DF['StudyID'] == patient_ID_XNAT]
        patient_ID = rows.iloc[0]['PatientID']
    studies = all_studies_DF.loc[all_studies_DF['PatientID'] == patient_ID]
    #studies.loc[:,'StudyDate'] =studies.loc[:,'StudyDate'].apply(pd.to_numeric) //FIX ordering not working
    #studies.sort_values(by='StudyDate', ascending=True)
    return studies

def generateExcludedImageList():
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + all_info_studies_file , sep = ';' , header = 0)

    #By Modality: exclude CT or None
    exclude = all_studies_DF[all_studies_DF['Modality'].isin(['None', 'CT'])]
    exclude['ReasonToExclude'] = 'CT'

    #By Series Description: exclude
    series_description = ['Costillasobl812','Costillasobl17','CostillasAP812','CostillasAP17','LatHorizontal','None']
    images = all_studies_DF[all_studies_DF['SeriesDescription'].isin(series_description)]
    images['ReasonToExclude'] = 'SeriesDescription'
    exclude= exclude.append(images)

    #By Protocol Name: exclude
    protocol_name = ['CHEST1VIE','Costillas', 'None','TORAXABDOMENThorax','CHEST1VIECHEST1VIEABDOMEN2VIES']
    images = all_studies_DF[all_studies_DF['ProtocolName'].isin(protocol_name)]
    images['ReasonToExclude'] = 'ProtocolName'
    exclude= exclude.append(images)

    #By Photometric Interpretation: exclude. Reason: many of them are DICOM annotation error, so this is necessary to reliable identify them to invert them in the preprocesing
    photometric_interpretation = ['MONOCHROME1']
    images = all_studies_DF[all_studies_DF['PhotometricInterpretation'].isin(photometric_interpretation)]
    images['ReasonToExclude'] = 'MONOCHROME1'
    exclude= exclude.append(images)

    #Images without reports
    images = all_studies_DF[all_studies_DF['Report'] == 'None']
    images['ReasonToExclude'] = 'NoReport'
    exclude= exclude.append(images)
    nr = '\n'.join(images.groupby('StudyID').groups.keys())
    f = open("Studies_without_reports.csv", 'w')
    f.write(nr)
    f.close()

    #Images that have failed to be preprocesses and are not in image_dir_preprocessed
    #i = all_studies_DF.ImagePath.apply(lambda x :  not os.path.exists(root + '/SJ' + x))
    images = all_studies_DF[ all_studies_DF.ImagePath.apply(lambda x :  not os.path.exists(root + '/SJ' + x))]
    images.to_csv("Not_processed_images.csv")
    images['ReasonToExclude'] = 'ImageNotProcessed'
    exclude= exclude.append(images)

    exclude.to_csv("Excluded_images_redundant.csv")
    exclude.groupby('ImagePath').first().to_csv("Excluded_images.csv")
    return exclude



def preprocess_images( study_ids = None):
    path = root + '/SJ'

    ConstPixelDims = None
    ConstPixelSpacing = None
    for i, study_id in enumerate(study_ids):
        images = getDicomInfo(study_id)
        for image in images.items():
            RefDs = image[1]
            filename = image[0]
            try:
                # Load dimensions based on the number of rows, columns
                ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns))
                # Load spacing values (in mm)
                if hasattr(RefDs, 'PixelSpacing'):
                    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))
                    x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
                    y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
                # The array is sized based on 'ConstPixelDims'
                ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
                # store the raw image data
                ArrayDicom[:, :] = RefDs.pixel_array
                #img = 1.0 - ArrayDicom * 1. / 4096
                #img = exposure.equalize_hist(img)
                #io.imsave(path + '/image_dir_processed/inverted_' + filename[-10:-4] + '.png', img)
                #Testing
                min = ArrayDicom.min()
                max = ArrayDicom.max()
                print ("ArrayDicom Mean: " + str(ArrayDicom.mean()))
                print ("Window Center: " + str(RefDs.WindowCenter))
                print ("ArrayDicom Min: " + str(min))
                print ("ArrayDicom Max: " + str(max))
                print ("Window Width: " + str(RefDs.WindowWidth))

                cp = np.copy(ArrayDicom)
                nmin = RefDs.WindowCenter - 0.5 - (RefDs.WindowWidth -1)/2
                nmax = RefDs.WindowCenter - 0.5 + (RefDs.WindowWidth -1)/2
                cp[ArrayDicom <= nmin] = min
                cp[ArrayDicom > nmax ] = max
                temp = ((ArrayDicom - (RefDs.WindowCenter - 0.5))/ (RefDs.WindowWidth -1) + 0.5) * (max-min) + min
                cp[ (ArrayDicom <= nmax) & (ArrayDicom > nmin)] = temp[(ArrayDicom <= nmax) & (ArrayDicom > nmin)]


                min = cp.min()
                max = cp.max()
                print ("ProcessedDicom Mean: " + str(cp.mean()))
                print ("ProcessedDicom Min: " + str(min))
                print ("ProcessedDicom Max: " + str(max))

                img =  cp * 1. / 4096
                #img = exposure.equalize_hist(img)
                io.imsave(path + '/image_dir_test/' + study_id + '_' + filename[-10:-4] + '.png', img)
                print ('Lung', i, filename)
            except:
                print ("Unexpected error:", sys.exc_info())

#Temporal method: generate a new file ("all_info_studies_st_file") where each row from all_info_study file is splitted in different rows (one for each sentence) adding the unsupervised label topic generated with para2vec ("source_topic_file")
def saveAllStudyTopicsFullDataset(save_file = all_info_studies_st_file, source_topic_file = None, study_file = all_info_studies_file):
    topics = '/sentence_clusters_100.csv' if not source_topic_file else source_topic_file
    path = root + '/Rx-thorax-automatic-captioning' + topics
    sent_topics = pd.read_csv(path, sep = ',',  header = 0, names =['key','class', 'ReportID','text'] )

    path = root + '/Rx-thorax-automatic-captioning' + study_file
    all_studies_DF = pd.read_csv(path, sep = ';' , header = 0)

    all_studies_DF.ReportID = all_studies_DF.ReportID.astype(str)
    sent_topics.ReportID = sent_topics.ReportID.astype(str)
    sent_topics['class'] = sent_topics['class'].astype(str)

    merge = pd.merge( all_studies_DF, sent_topics, how='left', on= 'ReportID')
    merge.to_csv(save_file)

#Temporal method: generate file for manual review of topics (review_sent_topic_#) where each row is a sentence with its  unsupervised topic and the count of occurences
#The total number of rows # is limited to first 1000 more frequent sentences
#The generated file is ready for manual review of topics in order to help add manually reviewed labels
def extract_sent_topics(save_file = extract_topics_file , source_topic_file = None ):
    topics = '/sentence_clusters.csv' if not source_topic_file else source_topic_file
    path = root + '/Rx-thorax-automatic-captioning' + topics
    sent_topics = pd.read_csv(path, sep = ',',  header = 0, names =['key','topic', 'ReportID','text'] )

    #unique sentences
    sent_topics.text  =  sent_topics.text.astype(str)
    unique_sentences = sent_topics.text.value_counts()
    print(unique_sentences.head())
    table = sent_topics.groupby(['text','topic'], as_index=False)
    df = pd.DataFrame(table.size(), columns = ['counts'])
    manual_review_sent_topic_1000 = df.sort_values(by = ['counts'], ascending = False)[:1000]
    manual_review_sent_topic_1000.to_csv('manual_review/review_sent_topic_1000.csv')

#Temporal method: generate file for manual review of topics (review_sent_topic_pending) where each row is a sentence with its  unsupervised topic and the count of occurences
#The total number of rows # is limited to sentences still not labeled (i.e those sentences in all_info_studies_st_file not labeled)
#The generated file is ready for manual review of topics
def pending_sent_labels(source_label_file = None):
    topics = '/manual_review/reviewed_sent_topic.csv' if not source_label_file else source_label_file
    path = root + '/Rx-thorax-automatic-captioning' + topics
    sent_labels = pd.read_csv(path, sep = ';',  header = 0 )
    sent_labels['labels']  = np.where(sent_labels['Unnamed: 4'].isnull(),sent_labels['Unnamed: 3'] ,sent_labels['Unnamed: 3'] + ',' +  sent_labels['Unnamed: 4'])

    all_labels = pd.Series([label for sent in sent_labels['labels'] for label in str(sent).split(',') ])
    all_labels.value_counts().sort_index().to_csv('manual_review/unique_labels.csv')

    path = root + '/Rx-thorax-automatic-captioning/' + all_info_studies_st_file
    all_studies_DF = pd.read_csv(path, sep = ',' , header = 0)
    merge = pd.merge( all_studies_DF, sent_labels, how='left', on= 'text')
    table = merge[merge['Unnamed: 3'].isnull()]
    table.text  =  table.text.astype(str)
    table = table.groupby(['text','class'], as_index=False)
    df = pd.DataFrame(table.size(), columns = ['counts'])
    df.to_csv('manual_review/review_sent_topic_100_pending.csv')


    print(merge.describe())

def merge_labeled_files():
    most_frequent_labeled_sentences = '/manual_review/reviewed_sent_topic_1000.csv'
    remaining_labeled_sentences = '/manual_review/reviewed_sent_topic_imgdataset.csv'
    path = root + '/Rx-thorax-automatic-captioning'
    sent_labels_mf = pd.read_csv(path + most_frequent_labeled_sentences, sep = ',',  header = 0 )
    sent_labels_rem = pd.read_csv(path + remaining_labeled_sentences, sep = ',',  header = 0 )
    sent_labels_rem.rename(columns={'class': 'topic'}, inplace=True)
    sent_labels = pd.concat([sent_labels_mf,sent_labels_rem], axis=0, ignore_index=True)
    sent_labels.rename(columns={'Unnamed: 3': '1','Unnamed: 4': '2','Unnamed: 5': '3','Unnamed: 6': '4','Unnamed: 7': '5','Unnamed: 8': '6', 'Unnamed: 9': '7','Unnamed: 10': '8','Unnamed: 11': '9'}, inplace=True)
    print(sent_labels.sample(5))
    column_names = ['text','topic', 'counts']
    column_names.extend(list('123456789'))
    sent_labels = sent_labels[column_names]
    sent_labels.to_csv('manual_review/labeled_sent_28K.csv')




#Temporal method: add to all info study file ("all_info_studies_file") the manual labels  ("source_label_file" e.g. labeled_sent_28K.csv)
def saveAllStudyLabelsFullDataset(source_label_file = None):

    path = root + '/Rx-thorax-automatic-captioning/' + all_info_studies_st_file
    all_studies_st_DF = pd.read_csv(path, sep = ',' , header = 0)

    sent_labels = '/manual_review/labeled_sent_28K.csv' if not source_label_file else source_label_file
    path = root + '/Rx-thorax-automatic-captioning' + sent_labels
    column_names = ['text','topic', 'counts']
    column_names.extend(list('123456789'))
    sent_labels = pd.read_csv(path, sep = ',',  header = 0, names =column_names )

    #add localization: To each sentence row in sent_labels add a new column named 0 with the localization info based on regex rules
    local_regex  = '/manual_review/localization_regex.csv'
    path = root + '/Rx-thorax-automatic-captioning' + local_regex
    local_regex = pd.read_csv(path, sep = ',',  header = 0, names= ['regex','lab'], dtype= {'lab': str} )
    def extract_matches(x):
        local_regex['matches'] = local_regex['regex'].apply(lambda pattern: ', '.join(re.findall(pattern, str(x))))
        m = ['loc ' + i for i in local_regex[local_regex['matches'] != '']['lab'].values]
        return m
    sent_labels['0'] = sent_labels['text'].apply(lambda x: extract_matches(x))
    # remove redundant localizations
    def remove_redundant(x):
        redundants = []
        for i in x:
            for j in x:
                if j != i and j  in i:
                    redundants.append(j)
        result = list(set(x) - set(redundants))
        return result
    sent_labels['0'] = sent_labels['0'].apply(lambda x: ','.join(remove_redundant(x)))
    sent_labels['label_and_local'] = sent_labels[list('1234567890')].apply(lambda x: ''.join(str(x.values )) , axis = 1)

    merge = pd.merge( all_studies_st_DF, sent_labels, how='left', on= 'text')
    merge.to_csv('manual_review/test.csv')


    path = root + '/Rx-thorax-automatic-captioning/' + all_info_studies_file
    all_studies_DF = pd.read_csv(path, sep = ';' , header = 0)

    groups = merge.groupby(['ImagePath'])
    def unique_labels(x):
        l = pd.unique(groups.get_group(x)[list('123456789')].values.ravel('K'))
        li = [x for x in l if pd.isnull(x) == False ]
        if len(li)>1 and 'normal' in li: #remove label 'normal' from studies with multiple labels
            if not (len(li)==2 and 'exclude' in li): #exception: keep normal if the only remaining label is exclude
                li.remove('normal')
        return li


    all_studies_DF['labels'] = all_studies_DF['ImagePath'].apply(lambda x:unique_labels(x) )

    def unique_by_group_and_column(x, column):
        l = pd.unique(groups.get_group(x)[[column]].values.ravel('K'))
        li = [x for x in l if x and pd.isnull(x) == False  ]
        return li

    all_studies_DF['localizations'] = all_studies_DF['ImagePath'].apply(lambda x:unique_by_group_and_column(x, '0') ) #Column 0 corresponds to the list of localizations by sentence
    all_studies_DF['study_label_and_local'] = all_studies_DF['ImagePath'].apply(lambda x:unique_by_group_and_column(x, 'label_and_local') )
    p = re.compile(r'[a-zA-Z\s\-]+')

    all_studies_DF['study_label_and_local'] = all_studies_DF['study_label_and_local'].apply(lambda x: [l.strip() for l in re.findall(p,str(x)) if l.strip() and 'nan' not in l and l.strip() != 'n' ])

    all_studies_DF.to_csv('manual_review/all_info_studies_labels.csv')

def summarizeAllStudiesLabel(all_info_studies_labels=None):
    study_labels_file = 'manual_review/all_info_studies_labels.csv' if not all_info_studies_labels else all_info_studies_labels

    path = root + '/Rx-thorax-automatic-captioning/' + study_labels_file
    all_studies_labels_DF = pd.read_csv(path, sep = ',' , header = 0)
    #count by study not by image (PA and Lateral image views share the same reports and therefore the same labels)
    row_labels = all_studies_labels_DF.groupby('StudyID').head(1)
    #count different labels
    labels = row_labels['labels'].astype('str')
    pattern = re.compile('[^a-zA-Z,\s\-]+')
    labels = [item.strip() for sublist in labels for item in pattern.sub('', sublist).split(",")]
    a = pd.Series(labels).value_counts()
    a.to_csv('manual_review/unique_labels_28K.csv')
    #count different localizations
    labels = row_labels['localizations'].astype('str')
    pattern = re.compile('[^a-zA-Z,\s\-]+')
    labels = [re.compile('loc\s+').sub('',item.strip()) for sublist in labels for item in pattern.sub('', sublist).split(",")]
    a = pd.Series(labels).value_counts()
    a.to_csv('manual_review/unique_localizations_28K.csv')


    return

def buildTreeCounts( ):
    stream = open("manual_review/trees.txt", "r")
    docs = yaml.load_all(stream)

    study_labels_file = 'manual_review/all_info_studies_labels.csv'
    path = root + '/Rx-thorax-automatic-captioning/' + study_labels_file
    all_studies_labels_DF = pd.read_csv(path, sep = ',' , header = 0)

    row_labels = all_studies_labels_DF.groupby('StudyID').head(1)
    row_labels['labels'] = row_labels['labels'].astype('str')
    pattern = re.compile('[^a-zA-Z,\s]+')
    row_labels['label_list'] = row_labels['labels'].apply(lambda r: set([item.strip() for item in pattern.sub('', r).split(",")]))


    study_localizations_count = 'manual_review/unique_localizations_28K.csv'
    study_labels_count = 'manual_review/unique_labels_28K.csv'
    path = root + '/Rx-thorax-automatic-captioning/' + study_labels_count
    study_labels_count_DF = pd.read_csv(path, sep = ',' ,names = ['label','counts'], dtype={'label': str, 'counts': np.int32})
    path = root + '/Rx-thorax-automatic-captioning/' + study_localizations_count
    study_localizations_count_DF = pd.read_csv(path, sep = ',' ,names = ['label','counts'], dtype={'label': str, 'counts': np.int32})
    study_labels_count_DF = pd.concat([study_labels_count_DF,study_localizations_count_DF], axis = 0)

    for doc in docs:
        tree_root = DictImporter().import_(doc)

        for pre, _, node in RenderTree(tree_root):
            try:
                if node.label == 'normal': #Normal if single label (studies with reports without pathological findings: [Normal + exclude], [Normal],  [Normal + Suboptimal]
                    s = {'normal','exclude','suboptimal study'}
                    node.study_counts = row_labels[row_labels['label_list'] <= s].shape[0]
                elif  node.label == 'exclude': #Exclude if single label (reports not informative, neither normal not pathological finding reported): [Exclude], [Exclude, suboptimal]
                    s = {'exclude','suboptimal study'}
                    node.study_counts = row_labels[row_labels['label_list'] <= s].shape[0]
                else:
                    node.study_counts = study_labels_count_DF.loc[study_labels_count_DF.label == node.label,'counts'].values[0]

            except:
                node.study_counts = 0

        with open("manual_review/tree_term_counts.txt", "a") as text_file:
            for pre, _, node in RenderTree(tree_root):
                node.study_node_counts = node.study_counts + sum(i.study_counts for i in node.descendants)
                print("%s%s %s" % (pre, node.label, node.study_node_counts), file=text_file)

        #with open("manual_review/tree_term_counts.txt", "a") as text_file:
            #print(RenderTree(tree_root), file=text_file)


        #labels (pathological findings) associated to differential diagnoses
        #load differential diagnosis subtree
        #for each diagnosis collect all other pathological findings in all_info_study_labels
        if tree_root.label == "differential diagnosis":
            for n in tree_root.descendants:
                n.ordered_near_labels = pd.Series([l for sublist in row_labels[row_labels['label_list'] >= {n.label}]['label_list'] for l in sublist ]).value_counts()

            #with open("manual_review/tree_term_counts.txt", "a") as text_file:
                #print(RenderTree(tree_root), file=text_file)
    return



merge_labeled_files()
saveAllStudyLabelsFullDataset()
summarizeAllStudiesLabel()
buildTreeCounts()




if ID_XNAT is not None:
    getAllInfo(ID_XNAT)


if modality is not None:
    summarizeAllStudiesDicomModality()


if patient_ID_XNAT is not None:
    print(getAllInfoPatient(patient_ID_XNAT =patient_ID_XNAT ))

if patient_ID is not None:
    print(getAllInfoPatient(patient_ID = patient_ID))

if filename is not None:
    saveAllStudyInfoFullDataset(filename)

if filename_st is not None:
    saveAllStudyTopicsFullDataset(filename_st, None)

if filename_to_describe is not None:
    summarizeAllStudies()

if categorical_field is not None:
    if categorical_field == 'all':
        fields = ['Modality','SeriesDescription','ProtocolName', 'BodyPartExamined','ViewPosition', 'CodeMeaning','PhotometricInterpretation', 'Manufacturer']
        for f in fields:
            summarizeAllStudiesByCategoricalField(categorical_field=f).to_csv(f + '.csv')
    else:
        print(summarizeAllStudiesByCategoricalField(categorical_field=categorical_field))

if numerical_field is not None:
    if numerical_field == 'all':
        fields = ['PatientBirth','StudyDate','Rows','Columns','PixelAspectRatio','SpatialResolution', 'XRayTubeCurrent','ExposureTime', 'ExposureInuAs','Exposure', 'RelativeXRayExposure', 'BitsStored', 'PixelRepresentation', 'WindowCenter', 'WindowWidth']
        for f in fields:
            print(summarizeAllStudiesByNumericalField( numerical_field = f))
    else:
        print(summarizeAllStudiesByNumericalField( numerical_field =numerical_field))

if split_images_side_front is not None:
    splitImagesFrontalSide()

if exclude is not None:
    generateExcludedImageList()

if imgs_ID_XNAT is not None:
    if imgs_ID_XNAT == 'test':
        imagePath = ['/image_dir_processed/259099525557219735264115148468152712554_m5ff9v.png',
                  '/image_dir_processed/299164937313584841767678964232362685010_sx5mth.png',
                  '/image_dir_processed/315752159734031831877330441630077004881-2_a83wu8.png']
        study_ids = []
        for image in imagePath:
            study_ids.append( re.search('image_dir_processed/(.+?)_',image).group(1))
        preprocess_images(study_ids)
    else:
        preprocess_images([imgs_ID_XNAT])

if sample_image is not None:
    generateMeanXRay(sample_image)

if extract_topics is not None:
    extract_sent_topics(extract_topics_file, None)