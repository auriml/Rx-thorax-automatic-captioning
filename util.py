#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
from skimage import io, exposure
import pydicom
import re


import os

import requests, zipfile, io
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Util methods')
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
parser.add_argument('-dc',  metavar='categorical_field', type=str, nargs=1,
                    help='summarize DICOM field categories for all studies. Default filename: /all_info_studies.csv. Possible categorical fields are '
                         'Modality;SeriesDescription;ProtocolName;ViewPosition;PhotometricInterpretation')
parser.add_argument('-dn',  metavar='numerical_field', type=str, nargs=1,
                    help='summarize DICOM numerical field  for all studies. Default filename: /all_info_studies.csv. Possible numerical fields are '
                         'Rows;Columns;PixelAspectRatio;SpatialResolution;RelativeXRayExposure')

parser.add_argument('-split',action='store_true',
                    help='split in side and front views based on DICOM info. Default source filename: /all_info_studies.csv.')

#parser.add_argument('u', metavar='username', type=str, nargs=1,
                    #help='XNAT account username')
#parser.add_argument('p', metavar='password', type=str, nargs=1,
                    #help='XNAT account password')
all_info_studies_file = '/all_info_studies.csv'
args = parser.parse_args()
ID_XNAT = args.s[0] if args.s  else  None
patient_ID_XNAT = args.ps[0] if args.ps  else  None
patient_ID = args.p[0] if args.p  else  None
modality = args.m if args.m  else  None
filename =  all_info_studies_file if args.f is None else None
filename_to_describe =  all_info_studies_file if args.d is None else None
categorical_field = args.dc if args.dc  else  None
numerical_field = args.dn if args.dn  else  None
split_images_side_front = args.split if args.split  else  None


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
    exp_id = dataset_asoc_DF.loc[dataset_asoc_DF[' MR ID XNAT '] == ID_XNAT]
    exp_id = exp_id.iloc[0][' Access Number ']
    study_info['codigoinforme'] = int(str(exp_id)[-7:])

    #access report
    filename = '/report_sentences_preprocessed.csv'
    path = root + '/Rx-thorax-automatic-captioning' + filename
    report_DF = pd.read_csv(path, sep = ',', encoding='ISO-8859-1') if  report_DF is None else report_DF
    text = report_DF.loc[report_DF['codigoinforme'] == study_info['codigoinforme']]
    try:
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
        images_info['ProtocolName'] = pattern.sub('', value.ProtocolName) if 'ProtocolName' in value else None #//TODO: summarize values, e.g. Trax ??
        images_info['ViewPosition'] = pattern.sub('', value.ViewPosition) if 'ViewPosition' in value else None #//TODO: summarize values
        images_info['Rows'] = value.Rows if 'Rows' in value else None
        images_info['Columns'] = value.Columns if 'Columns' in value else None
        images_info['PixelAspectRatio'] = value.PixelAspectRatio if 'PixelAspectRatio' in value else None
        images_info['SpatialResolution'] = value.SpatialResolution if 'SpatialResolution' in value else None
        images_info['PhotometricInterpretation'] = pattern.sub('', value.PhotometricInterpretation)  if 'PhotometricInterpretation' in value else None#//TODO check differences between MONOCHROME2 and MONOCHROME1
        images_info['RelativeXRayExposure'] = value.RelativeXRayExposure if 'RelativeXRayExposure' in value else None
        images.append(images_info)

    return images

def saveAllStudyInfoFullDataset(save_file = all_info_studies_file, dataset_asoc_file = '/dataset_asoc_abril18.csv'):
    dataset_asoc = '/dataset_asoc_abril18.csv' if not dataset_asoc_file else dataset_asoc_file
    path = root + '/Rx-thorax-automatic-captioning' + dataset_asoc
    dataset_asoc_DF = pd.read_csv(path, sep = ';', )

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
        studyID = r[' MR ID XNAT ']
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

    #Side view images where StudyDescription is in lat array (a manual selection from values of StudyDescription,
    # please run summarizeAllStudiesByCategory when new images are added to dataset to identify new values)
    lat = ["Lateral","Lateralizq", "LatVertical", "LatHorizontal", "Decblatizq"]
    side_images = all_studies_DF[all_studies_DF['SeriesDescription'].isin(lat)]

    #Front view images where StudyDescription is in front array (a manual selection from values of StudyDescription,
    # please run summarizeAllStudiesByCategory when new images are added to dataset to identify new values)
    front = ["Trax","Tórax","PA", "PAhoriz","APhorizontal","PAvertical", "pulmon", "AP","torax","APhoriz", "APvertical",
    "Lordtica", "APHorizontal", "PAHorizontal","Pediatra3aos", "Pediatría3años","APVertical", "Pedit3aos",  "Pediát3años", "W033TraxPA"]
    front_images = all_studies_DF[all_studies_DF['SeriesDescription'].isin(front)]

    side_images.to_csv('side_images.csv')
    front_images.to_csv('front_images.csv')

    return {'side' : side_images, 'front': front_images }

def summarizeAllStudiesByCategory (file = all_info_studies_file, byCategory = None):
    #Return dictionary where each key is one class and each value is a tuple with number of ocurrences and one example
    dict = {}
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + file, sep = ';' , header = 0)
    n = all_studies_DF.groupby([byCategory]).count()

    return n
def summarizeAllStudies(file = all_info_studies_file):
    summary = {}
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + file, sep = ';' , header = 0)

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
    plt.savefig('StudiesPerPatientHistogram.pdf')
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
    plt.savefig('BirthYearHistogram.pdf')  # saves the current figure
    plt.gcf().clear()

    #Age year  (mean, min, max), (distribution by year)
    print ('Age Distribution: ')
    study_dates = all_studies_DF['StudyDate'].apply(lambda x: x[:4])
    ages = pd.to_numeric(study_dates, errors='coerce' ) -  years
    print (ages.describe())
    ages.hist()
    plt.title("Age Histogram (n = " + str(n) + ")" )
    plt.xlabel("Patient's Age")
    plt.ylabel("Images")
    plt.savefig('AgeHistogram.pdf')
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
    plt.savefig('StudyYearHistogram.pdf')
    plt.gcf().clear()


    months = pd.DatetimeIndex(study_dates).month.values
    month_names = ['JAN', 'FEB', 'MAR', 'APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    freq = pd.Series(months).value_counts().sort_index()
    freq.plot(kind='bar')
    plt.title("Study Month Histogram (n = " + str(n) + ")" )
    plt.xlabel("Month of Study")
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], month_names)
    plt.ylabel("Images")
    plt.savefig('StudyMonthHistogram.pdf')
    plt.gcf().clear()



    #Modality (distribution by modality) //TODO: remove CT aprox 800
    print('Study Modality Distribution')
    modalities = all_studies_DF['Modality'].value_counts()
    print(modalities)
    #Series Description (distribution) //TODO:
# remove residuals:
# Costillasobl812       4
# LatVertical           4
# Costillasobl17        3
# PAHorizontal          3
# APHorizontal          3
# CostillasAP812        3
# CostillasAP17         2
# LatHorizontal         2
# Decblatizq            1
# Pediatra3aos          1
# APVertical            1
# W033TraxPA            1
# Pediát3años           1
# Pediatría3años        1
# None                  1
# Pedit3aos             1
    #Modality (distribution by modality)
    print('Study Description Distribution')
    modalities = all_studies_DF['SeriesDescription'].value_counts()
    print(modalities)

    #Images without reports
    no_reports = all_studies_DF[all_studies_DF['Report'] == 'None']
    print ("Studies with no reports:")
    nr = '\n'.join(no_reports.groupby('StudyID').groups.keys())
    f = open("Studies_without_reports.csv", 'w')
    f.write(nr)
    f.close()

    #Radiation exposure levels by study year
    print("Relative Radiation exposure levels")


    #Radiation exposure levels by type of Rx (Lateral vs AP)

    return summary

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

if ID_XNAT is not None:
    getAllInfo(ID_XNAT)


if modality is not None:
    summarizeAllStudiesDicomModality()


if patient_ID_XNAT is not None:
    print(getAllInfoPatient(patient_ID_XNAT =patient_ID_XNAT ))

if patient_ID is not None:
    print(getAllInfoPatient(patient_ID = patient_ID))

if filename is not None:
    saveAllStudyInfoFullDataset(filename_to_describe)

if filename_to_describe is not None:
    summarizeAllStudies()

if categorical_field is not None:
    summarizeAllStudiesByCategory(file = all_info_studies_file, category = categorical_field)

if split_images_side_front is not None:
    splitImagesFrontalSide()