#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
from skimage import io, exposure, transform, img_as_float
import pydicom
import re


import pathlib

import requests, zipfile
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import re
import yaml
from anytree import AnyNode
from anytree.exporter import DictExporter
from anytree.importer import DictImporter
from pprint import pprint  # just for nice printing
from anytree import RenderTree , search # just for nice printing
import remotedebugger as rd


parser = argparse.ArgumentParser(description='Util methods')
#rd.attachDebugger(parser)
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
                    help='split in type of projections views based on manual review of DICOM info and ResNet50.')

parser.add_argument('-e',action='store_true',
                    help='generate list of images to exclude. It excludes non Rx thorax studies or if patient position is not vertical or if image has not associated report. Default source filename: /all_info_studies.csv.')

parser.add_argument('-imgs', metavar='MR_ID_XNAT', type=str, nargs=1,
                    help='preprocess all images of a study identified with MR ID XNAT')

parser.add_argument('-imgm', action='store_true',
                    help='generate mean X-Ray picture and mean standard deviation picture for each position')
parser.add_argument('-fst',  metavar='filename', type=str, nargs='?', default= True,
                    help='save in filename all info for all studies by sentence topic. Default filename: /all_info_studies_sent_topics.csv  ')
parser.add_argument('-est',  metavar='filename', type=str, nargs='?', default= True,
                    help='extract and save in filename sentence topics. Default filename: /extract_sent_topics.csv  ')

parser.add_argument('-public', action='store_true', help='generate final File with all info to be public')

#parser.add_argument('u', metavar='username', type=str, nargs=1,
                    #help='XNAT account username')
#parser.add_argument('p', metavar='password', type=str, nargs=1,
                    #help='XNAT account password')
#all_info_studies_file = '/all_info_studies_nonXNAT.csv'
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
solve_images_projection = args.split if args.split  else  None
exclude = args.e if args.e  else  None
imgs_ID_XNAT = args.imgs[0] if args.imgs  else  None
image_mean = args.imgm if args.imgm  else  None
extract_topics = extract_topics_file if args.est is None else None
save_public_file = args.public if args.public else None


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


def getDicomFields(value):
    pattern = re.compile('[\W_]+')
    images_info = {}
    images_info["PatientID"] = value.PatientName  if 'PatientName' in value else None #codified patient's name //TODO: use it to ensure that the same patient is not both in the training and testing sets to avoid overfitting
    images_info["PatientBirth"] = str(value.PatientBirthDate)[:4] if 'PatientBirthDate' in value else None
    images_info["PatientSex"] = value.PatientSex if 'PatientSex' in value else None
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
    return images_info

def getDicomInfoStudy(ID_XNAT):
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
    dicoms = getDicomInfoStudy(ID_XNAT)
    
    for key, value in dicoms.items():
        print(value)
        images_info = getDicomFields(value)
        images_info['ReportID'] = study_info['codigoinforme']
        images_info['Report'] = study_info['report']
        images_info["ImagePath"] = key #TODO: image path should be the preprocessed image path not the dcm path
        images_info["StudyID"] = ID_XNAT #A study has 1 to Many images
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


def saveAllStudyInfoNonXNATDataset(save_file = '/all_info_studies_nonXNAT.csv', dataset_asoc_file = '/pacientes_bloque_nuevo_220518.txt'):
    dataset_asoc = dataset_asoc_file if dataset_asoc_file is not None else '/pacientes_bloque_nuevo_220518.txt' #Columns: "Access Number   Study instance UID      Id Anonimized"
                    
    all_studies_DF = None
    empty = False
    path = root + '/Rx-thorax-automatic-captioning' + save_file
    if os.path.exists(path):
        all_studies_DF = pd.read_csv(path, sep = ';' , header = 0)
    else:
        empty = True
    f = open(path, 'a')

    
    #Access each .dcm in anidated folder
    path = root + '/SJ'
    walk_path = path + '/salinas/' 
    for dirName, subdirList, fileList in os.walk(walk_path ):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                DICOMImagePath = dirName + "/" + filename
                if all_studies_DF is None or (DICOMImagePath not in all_studies_DF["DICOMImagePath"].values):
                    try:
                        RefDs = pydicom.read_file(os.path.join(dirName,filename),force=True)
                        image_info = getDicomFields(RefDs)
                        P_ID = RefDs.PatientID if 'PatientID' in RefDs else None #New image dataset does not have available XNAT ID, therefore PatientID from DICOM is used instead. Do not confuss with PatientName value 
                        ID_XNAT = RefDs.StudyInstanceUID if 'StudyInstanceUID' in RefDs else None
                        ID_XNAT = ID_XNAT.replace('.','')
                        image_info["StudyID"] = ID_XNAT #A xnat study has 1 to Many images, 
                        
                        #access number of report from DICOM
                        path = root + '/Rx-thorax-automatic-captioning' + dataset_asoc
                        if ('RequestAttributesSequence' in RefDs and 'RequestedProcedureID' in RefDs.RequestAttributesSequence[0] ):
                            report_id =  RefDs.RequestAttributesSequence[0].RequestedProcedureID  
                            image_info['ReportID'] = int(str(report_id)[-7:])
                        else: #If not in DICOM, try to retrieve it from association file
                            dataset_asoc_DF = pd.read_csv(path, sep = '\t', )
                            exp_id = dataset_asoc_DF.loc[dataset_asoc_DF['Study instance UID'] == ID_XNAT]
                            exp_id = int(exp_id.iloc[0]['Access Number'])
                            image_info['ReportID'] = int(str(exp_id)[-7:])
                        try:
                            #access report
                            filename = '/report_sentences_preprocessed.csv'
                            path = root + '/Rx-thorax-automatic-captioning' + filename
                            report_DF = pd.read_csv(path, sep = ',', encoding='ISO-8859-1') 
                            text = report_DF.loc[report_DF['codigoinforme'] == image_info['ReportID']]
                            image_info['Report'] = text.iloc[0]['v_preprocessed']

                        except:
                            image_info['Report'] = None
                            print("MISSING REPORT: " +str(ID_XNAT) + ' ' + str(image_info['ReportID']))
                            pass

                        suffix_image_id = DICOMImagePath.replace('/','-')
                        image_info["ImagePath"] = '/image_dir_processed/' + str(ID_XNAT) + '_' + suffix_image_id[-14:-4] + '.png'
                        image_info["DICOMImagePath"] = DICOMImagePath
                        if empty:
                            f.write(";".join(image_info.keys()) + '\n')
                        empty = False
                        f.writelines(";".join(str(e) for e in image_info.values()) + '\n')

        
                        
                    except:
                        print('Failed opening DICOM file: ' + DICOMImagePath, sys.exc_info())
                        #raise
                        pass
                #else:
                    #print(DICOMImagePath + "skipped, already in file")

    f.close()
    return 

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

def generatePositionsFileToReview():
    dfa = pd.read_csv('all_info_studies_labels_160K.csv', header = 0)
    dfs = pd.read_csv('SJ_chest_x_ray_images_labels_160K.csv', header = 0)
    dfs['ImageID'] = dfs['ImageID'].astype(str)

    dfa['ImageID'] = dfa['ImagePath'].str.split("/").str.get(-1)
    dfa['ImageID'] = dfa['ImageID'].astype(str)

    m = pd.merge(dfa,dfs,how='left', on='ImageID')
    m = m[['Projection','ViewPosition','CodeMeaning','ProtocolName','SeriesDescription', 'ImageID']].astype(str)
    c = m.groupby(['Projection','ViewPosition','ProtocolName','SeriesDescription','CodeMeaning']).ImageID.count()
    c = pd.DataFrame(c)
    c.to_csv('Positions.csv')
    return 

def solve_images_projection():
    dfa = pd.read_csv('_all_info_studies_labels_160K.csv', header = 0).astype(str)
    print(dfa.shape)
    #projections are resolved manually based on non-structured text in 'ViewPosition','CodeMeaning','ProtocolName','SeriesDescription']
    #those non resolvable based on those fields are marked as "UNK"
    #merge files
    pr = pd.read_csv('Positions_Reviewed.csv', header = 0).astype(str)
    m = pd.merge(dfa, pr, how='left', on=['ViewPosition','CodeMeaning','ProtocolName','SeriesDescription'])
    m.drop_duplicates(subset = 'ImagePath', inplace=True)
    dfa = pd.merge(dfa, m[['ImagePath','Review', 'Pediatric']], on='ImagePath', how='left')
    
    #UNK projections are then resolved in PA and L by pretrained model based on ResNet50
    #merge files, Review_x contains projections manually reviewed, Review_y contains projections classified by RESNET
    dfu = pd.read_csv('../chestViewSplit/all_info_studies_labels_projections_160K.csv', header = 0).astype(str)
    dfa = pd.merge(dfa, dfu[['ImagePath','Review']], on='ImagePath', how='left')
    dfa['MethodProjection'] = 'Manual review of DICOM fields'
    dfa.loc[dfa['Review_x'] == 'UNK','MethodProjection'] = 'resnet-50.t7'
    dfa.to_csv('all_info_studies_labels_projections_160K.csv')
    return


def summarizeAllStudiesByCategoricalField (file = all_info_studies_file, categorical_field = None):
    #Return dataframe where each row is one class and  values contains the number of ocurrences and one example
    #Possible categorical fields are: 'Modality;SeriesDescription;ProtocolName;ViewPosition;PhotometricInterpretation'
    #all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + file, sep = ';' , header = 0)
    all_info_studies_file_prefix = "/all_info_studies" 
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + all_info_studies_file_prefix + '_nonXNAT.csv', sep = ';' , header = 0)
    all_studies_DF = pd.concat([all_studies_DF,pd.read_csv(root + '/Rx-thorax-automatic-captioning' + all_info_studies_file_prefix + '.csv', sep = ';' , header = 0)], sort=False)
    all_studies_DF = all_studies_DF.drop_duplicates()
    n = all_studies_DF.groupby(categorical_field).first()
    c = all_studies_DF[categorical_field].value_counts()
    n = n.join(c)

    return n
def summarizeAllStudiesByNumericalField (file = all_info_studies_file, numerical_field = None):
    #Return dictionary where each key is one class and each value is a tuple with number of ocurrences and one example
    dict = {}
    #all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + file, sep = ';' , header = 0)
    all_info_studies_file_prefix = "/all_info_studies" 
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + all_info_studies_file_prefix + '_nonXNAT.csv', sep = ';' , header = 0)
    all_studies_DF = pd.concat([all_studies_DF,pd.read_csv(root + '/Rx-thorax-automatic-captioning' + all_info_studies_file_prefix + '.csv', sep = ';' , header = 0)], sort=False)
    all_studies_DF = all_studies_DF.drop_duplicates()
    #exclude non-evaluable images
    excluded = pd.read_csv('Excluded_images_nonXNAT.csv', sep = ',' , header = 0)
    excluded = pd.concat([excluded, pd.read_csv('Excluded_images.csv', sep = ',' , header = 0)],sort = False)
    print("Number of excluded studies: " + str(excluded.StudyID.nunique()))
    all_studies_DF = all_studies_DF[~all_studies_DF['ImagePath'].isin(excluded['ImagePath'].values)]
    idx = all_studies_DF[all_studies_DF['ImagePath'].isin(excluded['ImagePath'].values)].index.values
    #all_studies_DF = all_studies_DF.drop(idx)
    print("Number of excluded images: " + str(excluded.ImagePath.nunique()))
    n = all_studies_DF[numerical_field]
    n = pd.to_numeric(n, errors='coerce')


    return n.describe()
#summarizeAllStudiesByNumericalField(numerical_field= 'Rows')

def summarizeAllStudies(file = all_info_studies_file):
    summary = {}
    all_info_studies_file_prefix = "/all_info_studies" 
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + all_info_studies_file_prefix + '_nonXNAT.csv', sep = ';' , header = 0)
    all_studies_DF = pd.concat([all_studies_DF,pd.read_csv(root + '/Rx-thorax-automatic-captioning' + all_info_studies_file_prefix + '.csv', sep = ';' , header = 0)], sort=False)
    all_studies_DF = all_studies_DF.drop_duplicates()
    print("Number of total studies: " + str(all_studies_DF.StudyID.nunique()))
    print("Number of total images: " + str(all_studies_DF.ImagePath.nunique()))
    print("Number of total patients: " + str(all_studies_DF.PatientID.nunique()))
    #exclude non-evaluable images
    excluded = pd.read_csv('Excluded_images_nonXNAT.csv', sep = ',' , header = 0)
    excluded = pd.concat([excluded, pd.read_csv('Excluded_images.csv', sep = ',' , header = 0)],sort = False)
    print("Number of excluded studies: " + str(excluded.StudyID.nunique()))
    all_studies_DF = all_studies_DF[~all_studies_DF['ImagePath'].isin(excluded['ImagePath'].values)]
    idx = all_studies_DF[all_studies_DF['ImagePath'].isin(excluded['ImagePath'].values)].index.values
    #all_studies_DF = all_studies_DF.drop(idx)
    print("Number of excluded images: " + str(excluded.ImagePath.nunique()))
    print("Number of excluded patients: " + str(excluded.PatientID.nunique()))
    
    print(excluded.groupby('ImagePath').first().ReasonToExclude.value_counts())
    

    #Number of different images
    n = all_studies_DF['ImagePath'].count()
    print("Number of curated images: " + str(n))
    #Number of different patients
    np = all_studies_DF.groupby(['PatientID']).count()
    print("Number of  patients: " + str(len(np)))
    
    #ax = serie.hist()
    #fig = ax.get_figure()
    #fig.savefig('PatientNumberDistribution.pdf')
    #import matplotlib.pyplot as plt
    #s.hist()
    #plt.savefig('path/to/figure.pdf')  # saves the current figure

    
    #Number of studies per patient (mean, min, max)
    ns = all_studies_DF.groupby(['StudyID']).count()
    print("Number of curated  studies: " + str(len(ns)))
    #Number of images per study (mean, min, max)
    print("Distribution of # of images per study (mean, min, max): ")
    ns = all_studies_DF.groupby(['StudyID'])['ImagePath'].nunique()
    print(ns.describe())
    ns = all_studies_DF.groupby(['PatientID'])['StudyID'].nunique()
    print("Distribution of # of studies per patient (mean, min, max): ")
    print(ns.describe())
    print("Distribution of # of images per patient (mean, min, max): ")
    print(all_studies_DF.groupby(['PatientID'])['ImagePath'].nunique().describe())
    patients_with_multiple_studies= ns[ns > 1]
    patients_with_multiple_studies.hist()
    plt.title("Histogram of Patients with Multiple Studies (n = " + str(patients_with_multiple_studies.count()) + ")" )
    plt.xlabel("Number of Studies")
    plt.ylabel("Number of Patients")
    plt.savefig('graphs/StudiesPerPatientHistogram_all.png')
    plt.gcf().clear()
    patients_with_multiple_studies.to_csv('Patients_with_multiple_studies_all.csv')


    #Birth year (mean, min, max), (distribution by year)
    print ('Birth Year Distribution: ')
    years = all_studies_DF['PatientBirth']
    years = pd.to_numeric(years, errors='coerce')
    years[years  > 2017] = pd.np.nan #dataset is up to 2017, there are mistakes in the birth year showing the impossible year 2052 
    print (years.describe())
    years.hist()
    plt.title("Birth Year Histogram (n = " + str(n) + ")" )
    plt.xlabel("Patient's Birth Year")
    plt.ylabel("Images")
    plt.savefig('graphs/BirthYearHistogram_all.png')  # saves the current figure
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
    plt.savefig('graphs/AgeHistogram_all.png')
    plt.gcf().clear()

    #Sex
    print("Gender Histogram (n = " + str(n) + ")")
    sex = all_studies_DF['PatientSex'].dropna()
    print(sex.value_counts())
    sex = sex.where(sex.isin(['F','M']))
    sex.value_counts().plot(kind='bar')
    plt.title("Gender Histogram (n = " + str(n) + ")" )
    plt.xlabel("Gender")
    plt.ylabel("Images")
    plt.savefig('graphs/GenderHistogram_all.png')
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
    plt.savefig('graphs/AgesByGenderHistogram_all.png')
    plt.gcf().clear()

    pivot = ages_sex.reset_index().pivot(columns='PatientSex', values='PatientAge')
    pivot.boxplot(showfliers = True)
    plt.title("Gender Ages Boxplot (n = " + str(n) + ")" )
    plt.xlabel("Gender")
    plt.ylabel("Patient Age")
    plt.savefig('graphs/AgesByGenderBoxplot_all.png')
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
    plt.savefig('graphs/StudyYearHistogram_all.png')
    plt.gcf().clear()


    months = pd.DatetimeIndex(study_dates).month.values
    month_names = ['JAN', 'FEB', 'MAR', 'APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    freq = pd.Series(months).value_counts().sort_index()
    freq.plot(kind='bar')
    plt.title("Study Month Histogram (n = " + str(n) + ")" )
    plt.xlabel("Month of Study")
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], month_names)
    plt.ylabel("Images")
    plt.savefig('graphs/StudyMonthHistogram_all.png')
    plt.gcf().clear()


    #Study Type distribution (portatil ['RXSIMPLETORAXPORTATIL1OMASPROYECCIONES','RXTORAXCONPORTATIL'], pediatric ['TORAXPEDIATRICO,'RXTORAXPAYLATPED']
    print("Study Type Histogram (n = " + str(n) + ")")
    typeS = all_studies_DF['CodeMeaning']
    typeS.value_counts().plot(kind='bar')
    plt.title("Study Type Histogram  (n = " + str(n) + ")" )
    plt.xlabel("Study Type")
    plt.ylabel("Images")
    plt.savefig('graphs/TypeHistogram_all.png')
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
    pivot = temp.reset_index().pivot(columns='Year', values='Exposure')
    pivot.boxplot()
    plt.title("Radiation Exposure by Study Year  (n = " + str(n) + ")" )
    plt.xlabel("Years")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByYear_all.png')
    plt.gcf().clear()
    pivot.boxplot(showfliers=False)
    plt.title("Radiation Exposure by Study Year  (n = " + str(n) + ")" )
    plt.xlabel("Years")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByYear_NotOutliers_all.png')
    plt.gcf().clear()


    #Radiation exposure levels by type of Rx 
    p_cost = pd.read_csv('position_costal_images_nonXNAT.csv')
    p_cost = pd.concat([p_cost,pd.read_csv('position_costal_images.csv')],sort=False)
    p_cost['Position'] = 'COSTAL'

    p_AP_horizontal = pd.read_csv('position_frontAPHorizontal_images_nonXNAT.csv')
    p_AP_horizontal = pd.concat([p_AP_horizontal,pd.read_csv('position_frontAPHorizontal_images.csv')],sort=False)
    p_AP_horizontal['Position'] = 'AP_horizontal'

    p_frontAP = pd.read_csv('position_frontAP_images_nonXNAT.csv')
    p_frontAP = pd.concat([p_frontAP,pd.read_csv('position_frontAP_images.csv')],sort=False)
    p_frontAP['Position'] = 'AP'

    p_frontPA = pd.read_csv('position_frontPA_images_nonXNAT.csv')
    p_frontPA = pd.concat([p_frontPA,pd.read_csv('position_frontPA_images.csv')],sort=False)
    p_frontPA['Position'] = 'PA'

    p_pediatric = pd.read_csv('position_pediatric_images_nonXNAT.csv')
    p_pediatric = pd.concat([p_pediatric,pd.read_csv('position_pediatric_images.csv')],sort=False)
    p_pediatric['Position'] = 'PED'

    p_side_ver_lef = pd.read_csv('position_side_ver_lef_images_nonXNAT.csv')
    p_side_ver_lef = pd.concat([p_side_ver_lef,pd.read_csv('position_side_ver_lef_images.csv')],sort=False)
    p_side_ver_lef['Position'] = 'L'

    p_side_ver_rig = pd.read_csv('position_side_ver_rig_images_nonXNAT.csv')
    p_side_ver_rig = pd.concat([p_side_ver_rig,pd.read_csv('position_side_ver_rig_images.csv')],sort=False)
    #p_side_ver_rig['Position'] = 'side_ver_rig' #lateral right position are not identificable by any field and therefore they are aggregated in a single group with the left sided x-ray
    p_side_ver_rig['Position'] = 'L'

    p = [p_cost, p_AP_horizontal, p_frontAP,p_side_ver_rig, p_side_ver_lef, p_pediatric,p_frontPA ]
    position = pd.concat(p)
    print(position.shape)
    print(position.columns)
    

    print("Position Views")
    print(position.Position.describe())
    n_by_position = position.groupby('Position').apply(lambda x: x.count())
    print(n_by_position)
    position['Exposure'] = pd.to_numeric(position['Exposure'], errors='coerce', downcast='integer' )

    pivot = position.reset_index().pivot(columns='Position', values='Exposure')
    pivot.boxplot()
    plt.title("Radiation Exposure by Position View")
    plt.xlabel("Position View")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByPositionView_all.png')
    plt.gcf().clear()

    pivot = position.reset_index().pivot(columns='Position', values='Exposure')
    pivot.boxplot(showfliers = False)
    plt.title("Radiation Exposure by Position View")
    plt.xlabel("Position View")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByPositionViewWithoutOutlayer_all.png')
    plt.gcf().clear()

    position = pd.concat([position, pd.read_csv('position_toreview_images_nonXNAT.csv'),pd.read_csv('position_toreview_images.csv')], axis = 0)
    position.Position.ix[position['Position'].isna()] = 'unk'
    position.to_csv('position_view_all.csv', columns = ['Position','ImagePath'])
    print(position.Position.describe())
 
    position.Position.value_counts().plot(kind='bar')
    plt.title("Position View Histogram  (n = " + str(n) + ")" )
    plt.xlabel("Position View")
    plt.ylabel("Images")
    plt.savefig('graphs/PositionViewHistogram_all.png')
    plt.gcf().clear()

    #RadiationExposureByModality (Digital Rx - DX vs Computer Rx - CR)
    print("Radiation exposure levels by modality: Digitalized Rx (DX) vs Computer Rx (CR)")
    temp = pd.DataFrame()
    temp['Modality'] = all_studies_DF['Modality']
    temp['Exposure'] = pd.to_numeric(all_studies_DF['Exposure'], errors='coerce', downcast='integer' )
    print(temp['Exposure'].describe())
    temp= temp.dropna()
    temp = temp[temp['Exposure'] > 0]
    pivot = temp.reset_index().pivot(columns='Modality', values='Exposure')
    pivot.boxplot()
    plt.title("Radiation Exposure by Modality  (n = " + str(n) + ")" )
    plt.xlabel("Modality")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByModality_all.png')
    plt.gcf().clear()
    pivot.boxplot(showfliers=False)
    plt.title("Radiation Exposure by  Modality  (n = " + str(n) + ")" )
    plt.xlabel("Years")
    plt.ylabel("Radiation Exposure (mAs)")
    plt.savefig('graphs/RadiationExposureByModality_NotOutliers_all.png')
    plt.gcf().clear()

    #Dynamic Range Distribution by type of Rx (Lateral vs AP)

    return summary
#summarizeAllStudies()
def generateMeanAndStdImage(d, position,n_sample):
    mean = np.full((400, 400), 0)
    for path in d:
        path = root + '/SJ' + path 
        try:
            img = img_as_float(io.imread(path))
            img = transform.resize(img, (400,400))
            #img = np.expand_dims(img, -1)
            img = np.uint8(img * 255)

            #concatenated_img = np.concatenate((concatenated_img,img), axis=2)
            mean = img + mean 
        except:
            print(path)
            pass

    mean =  mean / n_sample
    img = exposure.equalize_hist(np.uint8(mean))
    io.imsave('graphs/_' + position + '_MeanImage.png',img)

    #calculate std
    std = np.full((400, 400), 0)
    for path in d:
        path = root + '/SJ' + path 
        try:
            img = img_as_float(io.imread(path))
            img = transform.resize(img, (400,400))
            #img = np.expand_dims(img, -1)
            img = np.uint16(img * 255)

            #concatenated_img = np.concatenate((concatenated_img,img), axis=2)
            std = np.power((img - mean),2) + std 
        except:
            print(path)
            pass
    std = np.sqrt(std/n_sample)
    img = exposure.equalize_hist(std)
    io.imsave('graphs/_' + position + '_StdImage.png',img)

    return
def generateMeanAndStdImageByEachPosition(n_sample = 500):
    df = pd.read_csv('all_info_studies_labels_160K.csv', header = 0)
    print(df.Review.value_counts())
    
    for position in (set(df.Review.value_counts().index.values) - set(['EXCLUDE'])):
        d = df[(df.Review == position) & (df.Pediatric != 'PED' )]
        d = d.sample(n = n_sample,random_state=3)
        d = d.ImagePath.values
        generateMeanAndStdImage(d,position, n_sample)
    #pediatric mean and std image
    print(df.Pediatric.value_counts())
    d = df[(df.Review != 'L') & (df.Pediatric == 'PED' )]
    d = d.ImagePath.values
    generateMeanAndStdImage(d,'PED', d.shape[0])


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

def generateExcludedImageList(fileSuffix = ".csv"): #fileSuffix is used to choose both the source and save file names.
    #possible values: _nonXNAT.csv, .csv
    all_info_studies_file_prefix = "/all_info_studies" 
    all_info_studies_file = all_info_studies_file_prefix + fileSuffix
    all_studies_DF = pd.read_csv(root + '/Rx-thorax-automatic-captioning' + all_info_studies_file , sep = ';' , header = 0)

    #By Modality: exclude CT or None
    exclude = all_studies_DF[all_studies_DF['Modality'].isin(['None', 'CT'])]
    exclude.loc[:,'ReasonToExclude'] = 'CT'

    #By BodyPartExamined [‘ABDOMEN’, ‘LOWEXM’, ‘RXESQUELETOPER’,’SERIEOSEA’]
    images = all_studies_DF[all_studies_DF['BodyPartExamined'].isin(['ABDOMEN', 'LOWEXM', 'RXESQUELETOPER','SERIEOSEA'])]
    images.loc[:,'ReasonToExclude'] = 'BodyPartExamined'
    exclude= exclude.append(images)

    #By Series Description: exclude
    series_description = ['LatHorizontal','None', 'Oblicuo', 'Transtorácico']
    images = all_studies_DF[all_studies_DF['SeriesDescription'].isin(series_description)]
    images.loc[:,'ReasonToExclude'] = 'SeriesDescription'
    exclude= exclude.append(images)

    #By Protocol Name: exclude
    protocol_name = ['CHEST1VIE','None','TORAXABDOMENThorax','CHEST1VIECHEST1VIEABDOMEN2VIES', 'Esternón', 'Húmero', 'Sternón']
    images = all_studies_DF[all_studies_DF['ProtocolName'].isin(protocol_name)]
    images.loc[:,'ReasonToExclude'] = 'ProtocolName'
    exclude= exclude.append(images)

    #By Photometric Interpretation: exclude. Reason: many of them are DICOM annotation error, so this is necessary to reliable identify them to invert them in the preprocesing
    photometric_interpretation = ['MONOCHROME1']
    images = all_studies_DF[all_studies_DF['PhotometricInterpretation'].isin(photometric_interpretation)]
    images.loc[:,'ReasonToExclude'] = 'MONOCHROME1'
    exclude= exclude.append(images)

    #Images without reports
    print("excluding images without reports..")
    images = all_studies_DF[all_studies_DF['Report'] == 'None']
    images.loc[:,'ReasonToExclude'] = 'NoReport'
    exclude= exclude.append(images)
    print(exclude.shape)
    nr = '\n'.join(images.groupby('StudyID').groups.keys())
    f = open("Studies_without_reports" + fileSuffix, 'w')
    f.write(nr)
    f.close()

    #Images that have failed to be preprocesses and are not in image_dir_preprocessed
    #i = all_studies_DF.ImagePath.apply(lambda x :  not os.path.exists(root + '/SJ' + x))
    print("excluding non processed images..")
    image_dir_processed = pathlib.Path(root + '/SJ'+ '/image_dir_processed')
    savedImagePaths = list(image_dir_processed.glob("*.png"))
    images = all_studies_DF[~all_studies_DF['ImagePath'].isin([ '/image_dir_processed/'+ it.name  for it in savedImagePaths])]
            
   

    #images = all_studies_DF[ all_studies_DF.ImagePath.apply(lambda x :  not os.path.exists(root + '/SJ' + x))]
    images.to_csv("Not_processed_images" + fileSuffix)
    images.loc[:,'ReasonToExclude'] = 'ImageNotProcessed'
    exclude= exclude.append(images)

    #Images with less than expected number of rows or columns
    print("excluding images with less than expected number of rows or columns..")
    columns = pd.to_numeric(all_studies_DF.Columns, errors='coerce')
    rows = pd.to_numeric(all_studies_DF.Rows, errors='coerce')
    images = all_studies_DF[(columns < 500) | (rows < 500)]
    print(images.shape[0])
    images.loc[:, 'ReasonToExclude'] = 'LowRowsOrColumns'
    exclude= exclude.append(images)

    exclude.to_csv("Excluded_images_redundant" + fileSuffix)
    exclude.groupby('ImagePath').first().to_csv("Excluded_images" + fileSuffix)
    return exclude



def preprocess_images( study_ids = None):
    path = root + '/SJ'

    ConstPixelDims = None
    ConstPixelSpacing = None
    for i, study_id in enumerate(study_ids):
        images = getDicomInfoStudy(study_id)
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

#Temporal method: generate a new file ("all_info_studies_st_file") where each row from all_info_study file is splitted in different rows (one for each sentence) 
# adding the unsupervised label topic generated with para2vec ("source_topic_file")
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

#Temporal method: generate file for manual review of topics (review_sent_topic_pending) 
# where each row is a sentence with its  unsupervised topic and the count of occurences
#The total number of rows # is limited to sentences still not labeled 
# (i.e those sentences in all_info_studies_st_file not labeled)
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


def extract_loc_matches(x, local_regex):
    local_regex['matches'] = local_regex['regex'].apply(lambda pattern: ', '.join(re.findall(pattern, str(x))))
    m = ['loc ' + i.strip() for i in local_regex[local_regex['matches'] != '']['lab'].values]
    return m
    
    
def remove_redundant(x):
    redundants = []
    for i in x:
        for j in x:
            if j != i and j  in i:
                redundants.append(j)
    result = list(set(x) - set(redundants))
    return result

#Temporal method: add to all info study file ("all_info_studies_file") 
# the manual labels  ("source_label_file" e.g. labeled_sent_28K.csv)
def saveAllStudyManualLabelsDataset(source_label_file = None):

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
    
    sent_labels['0'] = sent_labels['text'].apply(lambda x: extract_loc_matches(x,local_regex)).apply(lambda x: remove_redundant(x))
    # remove redundant localizations
    
    sent_labels['label_and_local'] = sent_labels[list('123456789')].apply(lambda x: [i for i in x.dropna().values]   , axis = 1)
    sent_labels['label_and_local'] = pd.Series(sent_labels['label_and_local'])  + pd.Series(sent_labels['0'])

    merge = pd.merge( all_studies_st_DF, sent_labels, how='left', on= 'text')
    merge.to_csv('manual_review/test.csv')


    path = root + '/Rx-thorax-automatic-captioning/' + all_info_studies_file
    all_studies_DF = pd.read_csv(path, sep = ';' , header = 0)

    groups = merge.groupby(['ImagePath'])
    def unique_labels(x): 
        serie = groups.get_group(x)[list('123456789')].values.ravel('K')
        serie = [re.sub(r' right side| left side| both side.*| bilateral', '', x) for x in serie if pd.isnull(x) == False] #remove localizations that were manually added to labels
        l = pd.unique(serie)
        li = [x for x in l if pd.isnull(x) == False ]
        if len(li)>1 and 'exclude' in li:
            li.remove('exclude') #first: remove label 'exclude' from studies with multiple labels
        if len(li)>1 and 'normal' in li: #second: remove label 'normal' from studies with multiple labels
            #if not (len(li)==2 and 'unchanged' in li): #exception: keep normal if the only remaining label is exclude
            li.remove('normal')
        return li

    all_studies_DF['labels'] = all_studies_DF['ImagePath'].apply(lambda x:unique_labels(x) )

    def agregate(x, column, unique = False):
        li = None
        try:
            if unique:
                li = list(set([item for sublist in groups.get_group(x)[[column]].values.ravel('K').tolist() for item in sublist ]))
            else:
                li = [item for sublist in groups.get_group(x)[[column]].values for item in sublist ]
        except:
            li = []
        return li

    all_studies_DF['localizations'] = all_studies_DF['ImagePath'].apply(lambda x:agregate(x, '0', True)) #Column 0 corresponds to the list of localizations by sentence
    all_studies_DF['study_label_and_local'] = all_studies_DF['ImagePath'].apply(lambda x: agregate(x, 'label_and_local', False ))
    
   
    
    all_studies_DF.to_csv('manual_review/all_info_studies_labels.csv')
    return



#Reusable method: generate  all_info_study_labels file e.g "all_info_study_labels_nonXNAT.csv" 
# from automatically labeled sentences e.g "sentences_reports_aut_labeled.csv" as outputted by the model
# columns: ,codigoinforme,text,labels_x,labels_y,labels_x,labe… (where only labels_x contains output labels as e.g. "('cardiomegaly', 'heart insufficiency')")
def saveAllStudyAutLabelsDataset(source_label_file = "sentences_reports_aut_labeled.csv", fileSuffix = "_nonXNAT.csv"):
    path = root + '/Rx-thorax-automatic-captioning/' + source_label_file
    #Load sentences automatically labeled
    df = pd.read_csv(path, sep = ',' ,dtype = str)
    df = df[(df.codigoinforme.astype(int) > 3200000) & (df.codigoinforme.astype(int) < 4750000)]
    
    #df = df[(df.codigoinforme.astype(int) == 4498858)]
    
    
    print("#Step 1: create field labels") 
    df['labels'] = df[df.columns[3:]].apply(lambda x: "".join(x.dropna().values),axis=1)
    df['labels'] = df.labels.str.replace(r'(,\)|[()\'\"])','').str.split(',')
    
    print("#Step 2: create field localizations from text")
    #add localization: To each sentence row in sent_labels add a new column named 0 with the localization info based on regex rules
    #and remove redundant localizations
    local_regex  = '/manual_review/localization_regex.csv'
    path = root + '/Rx-thorax-automatic-captioning' + local_regex
    local_regex = pd.read_csv(path, sep = ',',  header = 0, names= ['regex','lab'], dtype= {'lab': str} )
    df['localizations'] = df['text'].apply(lambda x: extract_loc_matches(x, local_regex)).apply(lambda x: remove_redundant(x))
    df['sent_label_and_local'] = df['labels'] + df['localizations']

    print("#Step 3: group sentences by study report and regenerate labels column removing redundant labels,") 
    # regenerate localizations column, 
    # generate study_label_and_local as sequence of sentence labels  and loc. 
    def unique_labels(x): 
        l = pd.unique(x)
        li = [x for x in l if pd.isnull(x) == False ]
        if len(li)>1 and 'exclude' in li:
            li.remove('exclude') #first: remove label 'exclude' from studies with multiple labels
        if len(li)>1 and 'normal' in li: #second: remove label 'normal' from studies with multiple labels
            #if not (len(li)==2 and 'exclude' in li): #exception: keep normal if the only remaining label is exclude
            li.remove('normal')
        return li
    df_reports = pd.DataFrame()
    df_reports[['labels','localizations', 'study_label_and_local']] = df.groupby('codigoinforme').apply(lambda x:  pd.Series([unique_labels([item for sublist in x['labels'].values.tolist() for item in sublist ]), 
    list(set([item for sublist in x['localizations'].values.tolist() for item in sublist ])),
    [item for sublist in x['sent_label_and_local'].values.tolist() for item in sublist ]]))
    df_reports['ReportID'] = df_reports.index.astype(int)

    print("Step 4: merge with all_info_studies_file") 
    all_info_studies_file_prefix = "/all_info_studies"
    path = root + '/Rx-thorax-automatic-captioning/' + all_info_studies_file_prefix + fileSuffix
    all_studies_DF = pd.read_csv(path, sep = ';' , header = 0)
    all_studies_DF.ReportID = all_studies_DF.ReportID.astype(int)
    merge = pd.merge( all_studies_DF, df_reports, how='left', on= 'ReportID')

    print("Step 5: save file all_info_studies_labels_nonXNAT.csv")
    print("Saving all_info_studies_labels")
    merge.to_csv('all_info_studies_labels'+ fileSuffix)
    return


   
def mergeAllStudyLabels160K():
    fileSuffix = ".csv"
    filePrefix = 'manual_review/' 
    study_labels_file = filePrefix  + 'all_info_studies_labels' + fileSuffix 
    path = root + '/Rx-thorax-automatic-captioning/' + study_labels_file
    manual_DF = pd.read_csv(path, sep = ',' , header = 0)
    manual_DF['MethodLabel'] = 'Physician'

    fileSuffix = "_nonXNAT.csv"
    filePrefix = '' 
    study_labels_file = filePrefix  + 'all_info_studies_labels' + fileSuffix 
    path = root + '/Rx-thorax-automatic-captioning/' + study_labels_file
    aut_DF = pd.read_csv(path, sep = ',' , header = 0)
    aut_DF['MethodLabel'] = 'RNN_model'

    merge =  pd.concat([manual_DF,aut_DF], axis = 0 ,ignore_index = True)
    print(merge.shape[0])
    merge.to_csv('all_info_studies_labels_160K.csv')
    

    return

def generatePublicFile():
    merge = pd.read_csv('all_info_studies_labels_projections_160K.csv', header = 0).astype(str)


    #exclude images
    #exclude non-evaluable images
    excluded = pd.read_csv('Excluded_images_nonXNAT.csv', sep = ',' , header = 0)
    excluded = pd.concat([excluded, pd.read_csv('Excluded_images.csv', sep = ',' , header = 0)],sort = False)
    print("Number of excluded studies: " + str(excluded.StudyID.nunique()))
    merge = merge[~merge['ImagePath'].isin(excluded['ImagePath'].values)]
    
    print("Number of excluded images: " + str(excluded.ImagePath.nunique()))
    print("Number of excluded patients: " + str(excluded.PatientID.nunique()))
    print(merge.shape[0])
    print(excluded.groupby('ImagePath').first().ReasonToExclude.value_counts())

    
    

    #generate birthYear derived field 
    merge['PatientBirth'] = pd.to_numeric(merge['PatientBirth'], errors='coerce')
    merge.loc[merge['PatientBirth']  > 2017,'PatientBirth'  ]= pd.np.nan #dataset is up to 2017, there are mistakes in the birth year showing the impossible year 2052 
    
 

    #generate public file
    num_zips = 50 
    new_DF = pd.DataFrame()
    new_DF['ImageID'] = merge['ImagePath'].str.split("/").str.get(-1)
    new_DF['ImageDir'] = pd.Series(50 * (merge.index +1) / merge.shape[0]).astype(int) 
    new_DF['StudyDate_DICOM'] = merge['StudyDate']
    new_DF['StudyID'] = merge['StudyID']
    new_DF['PatientID'] = merge['PatientID']
    new_DF['PatientBirth'] = merge['PatientBirth']
    new_DF['PatientSex_DICOM'] = merge['PatientSex']
    new_DF['ViewPosition_DICOM'] = merge['ViewPosition']
    new_DF['Projection'] = merge['Review_y']
    new_DF['MethodProjection'] = merge['MethodProjection'] 
    new_DF['Pediatric'] = merge['Pediatric']
    new_DF['Modality_DICOM'] = merge['Modality']
    new_DF['Manufacturer_DICOM'] = merge['Manufacturer']
    new_DF['PhotometricInterpretation_DICOM'] = merge['PhotometricInterpretation']
    new_DF['PixelRepresentation_DICOM'] = merge['PixelRepresentation']
    new_DF['PixelAspectRatio_DICOM'] = merge['PixelAspectRatio']
    new_DF['SpatialResolution_DICOM'] = merge['SpatialResolution']
    new_DF['BitsStored_DICOM'] = merge['BitsStored']
    new_DF['WindowCenter_DICOM'] = merge['WindowCenter']
    new_DF['WindowWidth_DICOM'] = merge['WindowWidth']
    new_DF['Rows_DICOM'] = merge['Rows']
    new_DF['Columns_DICOM'] = merge['Columns']
    new_DF['XRayTubeCurrent_DICOM'] = merge['XRayTubeCurrent']
    new_DF['Exposure_DICOM'] = merge['Exposure']
    new_DF['ExposureInuAs_DICOM'] = merge['ExposureInuAs']
    new_DF['ExposureTime'] = merge['ExposureTime']
    new_DF['RelativeXRayExposure_DICOM'] = merge['RelativeXRayExposure']
    new_DF['ReportID'] = merge['ReportID']
    new_DF['Report'] = merge['Report']
    new_DF['MethodLabel'] = merge['MethodLabel']
    new_DF['Labels'] = merge['labels']
    new_DF['Localizations'] = merge['localizations']
    new_DF['LabelsLocalizationsBySentence'] = merge['study_label_and_local']

    CUI_DF = pd.read_csv('manual_review/term_CUI_counts.txt', sep = ',', header = 0)
    CUI_DF['CUI'].dropna(inplace = True)

    label_CUI_DF = CUI_DF[CUI_DF['tree'].isin(['suboptimal study','differential diagnosis','radiological finding','exclude','normal'])].groupby('label').first() 
    loc_CUI_DF = CUI_DF[(CUI_DF['tree'] == 'localization') ].groupby('label').first()

    dla = label_CUI_DF.to_dict()['CUI']
    dlo = loc_CUI_DF.to_dict()['CUI']

    import ast
    def delabel(labels,d):
        r = None
        if labels is not None:
            try:
                r = pd.Series([d.get(label.replace('loc ','').strip(),'') for label in  ast.literal_eval(labels)]).dropna().values
            except:
                r = None
        return r
    #delabel(labels.__repr__())
    new_DF.loc[:,'labelCUIS'] = new_DF.loc[:,'Labels'].apply(lambda l: delabel(l,dla))
    new_DF.loc[:,'LocalizationsCUIS'] = new_DF.loc[:,'Localizations'].apply(lambda l: delabel(l,dlo))
    print(new_DF.shape[0])
    new_DF.drop_duplicates(subset = 'ImageID', inplace=True)
    print(new_DF.shape[0])

    new_DF.to_csv('SJ_chest_x_ray_images_labels_160K.csv')

    return
     


def summarizeAllStudiesLabel(all_info_studies_labels=None, fileSuffix = "_160K.csv"):
    filePrefix = 'manual_review/' if fileSuffix == ".csv" else ""
    study_labels_file = filePrefix  + 'all_info_studies_labels' + fileSuffix if not all_info_studies_labels else all_info_studies_labels
    
    path = root + '/Rx-thorax-automatic-captioning/' + study_labels_file
    all_studies_labels_DF = pd.read_csv(path, sep = ',' , header = 0)
    #count by study not by image (PA and Lateral image views share the same reports and therefore the same labels)
    row_labels = all_studies_labels_DF.groupby('StudyID').head(1)
    #count different labels
    labels = row_labels['labels'].astype('str')
    pattern = re.compile('[^a-zA-Z,\s\-]+')
    labels = [item.strip() for sublist in labels for item in pattern.sub('', sublist).split(",")]
    a = pd.Series(labels).value_counts()
    a.to_csv(filePrefix + 'unique_labels' + fileSuffix)
    #count different localizations
    labels = row_labels['localizations'].astype('str')
    pattern = re.compile('[^a-zA-Z,\s\-]+')
    labels = [re.compile('loc\s+').sub('',item.strip()) for sublist in labels for item in pattern.sub('', sublist).split(",")]
    a = pd.Series(labels).value_counts()
    a.to_csv(filePrefix +'unique_localizations' + fileSuffix)


    return

def buildTreeCounts(fileSuffix = "_160K.csv" ):
    stream = open("manual_review/trees_code.txt", "r")
    docs = yaml.load_all(stream)

    filePrefix = 'manual_review/' if fileSuffix == ".csv" else ""
    study_labels_file = filePrefix +'all_info_studies_labels' + fileSuffix
    path = root + '/Rx-thorax-automatic-captioning/' + study_labels_file
    all_studies_labels_DF = pd.read_csv(path, sep = ',' , header = 0)

    row_labels = all_studies_labels_DF.groupby('StudyID').head(1)
    row_labels['labels'] = row_labels['labels'].astype('str')
    pattern = re.compile('[^a-zA-Z,\s]+')
    row_labels['label_list'] = row_labels['labels'].apply(lambda r: set([item.strip() for item in pattern.sub('', r).split(",")]))


    study_localizations_count = filePrefix + 'unique_localizations' + fileSuffix
    study_labels_count = filePrefix + 'unique_labels' + fileSuffix
    path = root + '/Rx-thorax-automatic-captioning/' + study_labels_count
    study_labels_count_DF = pd.read_csv(path, sep = ',' ,names = ['label','counts'], dtype={'label': str, 'counts': np.int32})
    path = root + '/Rx-thorax-automatic-captioning/' + study_localizations_count
    study_localizations_count_DF = pd.read_csv(path, sep = ',' ,names = ['label','counts'], dtype={'label': str, 'counts': np.int32})
    study_labels_count_DF = pd.concat([study_labels_count_DF,study_localizations_count_DF], axis = 0)
    
    df = pd.DataFrame(columns = ['tree', 'label', 'CUI', 'label_count', 'branch_count'])
    i= 0
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

        
        for pre, _, node in RenderTree(tree_root):
            node.study_node_counts = node.study_counts + sum(i.study_counts for i in node.descendants)
            with open(filePrefix + "tree_term_CUI_counts" + fileSuffix, "a") as text_file:
                CUI = None
                try:
                    CUI = node.concept
                    m = re.search('\((.+?)\)', CUI)
                    if m:
                        CUI = m.group(1)

                    print("%s%s [CUI:%s, counts:%s, %s]\\\\" % (pre, node.label, CUI, node.study_counts, node.study_node_counts), file=text_file)
                except: 
                    print("%s%s [counts:%s, %s]\\\\" % (pre, node.label, node.study_counts, node.study_node_counts), file=text_file)
                    pass
            df.loc[i]=[tree_root.label, node.label, CUI,node.study_counts, node.study_node_counts]
            i +=1
        
                
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
    df.drop_duplicates(inplace=True)
    df.to_csv(filePrefix +'term_CUI_counts' + fileSuffix)
    return



#merge_labeled_files()
#saveAllStudyManualLabelsDataset()
#saveAllStudyAutLabelsDataset()
#mergeAllStudyLabels160K()
#generatePublicFile()
#summarizeAllStudiesLabel() #save graphs and generate file of derived field positionviews 
#summarizeAllStudies() #save unique labels files 
#buildTreeCounts()

def saveAllSentences():
    #load sentences from report_sentences_preprocessed
    #Generate a file (one sentence for each line) 
    textFile = "report_sentences_preprocessed_no_masa.csv" 
    path = root + '/Rx-thorax-automatic-captioning' + textFile
    df = pd.read_csv(textFile , keep_default_na=False)
    df['v_preprocessed'] = df['v_preprocessed'].str.replace(r'\.$','')
    sentences =  df['v_preprocessed'].str.split('\\.')
    sentences = sentences.apply(lambda x: [y.strip() for y in x if y.strip()])

    #Save file with reports splitted in sentences (sentences_preprocessed.csv) so that each sentence is a line and keep the report_id of each sentence
    cols = ['codigoinforme']
    df = pd.DataFrame({x: np.repeat(df[x], [len(y) for y in  sentences]) for x in cols})
    df['text'] = np.concatenate([x for x in sentences if x])
    df = df.reset_index(drop=True)
    df.to_csv('sentences_preprocessed.csv', index=False)

#saveAllSentences()
#saveAllStudyInfoNonXNATDataset()


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

if solve_images_projection is not None:
    solve_images_projection()

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

if image_mean is not None:
    generateMeanAndStdImageByEachPosition()

if extract_topics is not None:
    extract_sent_topics(extract_topics_file, None)

if save_public_file is not None:
    generatePublicFile()