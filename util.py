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

parser = argparse.ArgumentParser(description='Util methods')
parser.add_argument('-s', metavar='MR_ID_XNAT', type=str, nargs=1,
                    help='display DICOM info for all images of a study identified with MR ID XNAT')
parser.add_argument('-p', metavar='MR_ID_XNAT', type=str, nargs=1,
                    help='display DICOM info for all images of all studies available for a patient that has a study identified with MR ID XNAT')
parser.add_argument('-m', metavar='DICOM field modality',
                    help='summarize all values in dataset for the DICOM field modality')
parser.add_argument('-f',  metavar='filename', type=str, nargs='?', default= True,
                    help='save in filename all info for all studies. Default filename: /all_info_studies.csv  ')
parser.add_argument('-d',  metavar='filename', type=str, nargs='?', default= True,
                    help='describe all info for all studies. Default filename: /all_info_studies.csv  ')

#parser.add_argument('u', metavar='username', type=str, nargs=1,
                    #help='XNAT account username')
#parser.add_argument('p', metavar='password', type=str, nargs=1,
                    #help='XNAT account password')

args = parser.parse_args()
ID_XNAT = args.s[0] if args.s  else  None
patient_ID_XNAT = args.p[0] if args.p  else  None
modality = args.m if args.m  else  None
filename =  '/all_info_studies.csv' if args.f is None else None
filename_to_describe =  '/all_info_studies.csv' if args.d is None else None


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
        images_info['SeriesDescription'] = pattern.sub('', value.SeriesDescription) if 'SeriesDescription' in value else None#//TODO: summarize values, e.g. APhorizontal
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

def saveAllStudyInfoFullDataset(save_file = '/all_info_studies.csv', dataset_asoc_file = '/dataset_asoc_abril18.csv'):
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

def summarizeAllStudies(file = '/all_info_studies.csv'):
    summary = {}
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + file, sep = ';' , header = 0)

    #Number of different patients
    n = all_studies_DF.groupby(['PatientID']).count()
    print("Number of different patients :" + str(n))
    #Number of studies per patient (mean, min, max)
    #Age (mean, min, max), (distribution by year)
    #Number of images per study (mean, min, max)
    #Study date (distribution by year)
    #Modality (distribution by modality)
    #Series Description (distribution)
    #


    return summary

def getAllInfoPatient(patient_ID_XNAT):
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning/all_info_studies.csv' , sep = ';' , header = 0)
    rows = all_studies_DF.loc[all_studies_DF['StudyID'] == patient_ID_XNAT]
    patientID = rows.iloc[0]['PatientID']
    studies = all_studies_DF.loc[all_studies_DF['PatientID'] == patientID]
    return studies

if ID_XNAT is not None:
    getAllInfo(ID_XNAT)


if modality is not None:
    summarizeAllStudiesDicomModality()


if patient_ID_XNAT is not None:
    print(getAllInfoPatient(patient_ID_XNAT))

if filename is not None:
    saveAllStudyInfoFullDataset(filename_to_describe)

if filename_to_describe is not None:
    summarizeAllStudies()
