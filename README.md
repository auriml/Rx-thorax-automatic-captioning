# PADCHEST Dataset 

The PADCHEST dataset is a chest x-ray labeled dataset containing 160K high resolution images with their corresponding labeled reports. Its detailed description and labeling methods are described in [1].  
Use of the PadChest is free to all researchers. Researchers seeking to use the full Clinical Database must formally request access in http://bimcv.cipf.es/bimcv-projects/padchest/ . By requesting access the user agrees that (1) he/she will not share the data, (2) he/she will make no attempt to reidentify individuals. The PadChest, although de-identified, still contains information regarding the clinical care of patients, and must be treated with appropriate respect. For attribution, please cite as [1]. 

## Multilabel Annotation 

PadChest includes more than 160,000 images obtained from 67,000 patients
that were interpreted and reported by radiologists at San Juan Hospital (Spain) from 2009 to 2017, covering six different position
views, and including additional information on image acquisition and patient demography. Padchest’s labels were extracted from the
radiological reports applying a Natural Language Processing (NLP) annotation pipeline based on bidirectional
Long Short-Term Memory Networks (LSTM) with attention mechanisms which was used to label the 73% of the reports using a ground-truth that was curated manually by trained physicians. It resulted in the annotation of 19 differential diagnoses, 103 anatomic locations and 179 different radiological findings mapped onto the NLM standard Unified Medical Language System (UMLS) using controlled biomedical vocabulary unique identifiers (CUIs) that were further organized into semantic hierarchical concept trees. 

## COVID19 pandemic
Padchest was obtained before the COVID19 pandemic but is the only pre-pandemic open large-scale dataset that provides labels for different patterns of infiltrates, consolidations, pneumonia and ground glass opacities, including their anatomical localizations as needed to diagnose COVID 19 pneumonia. In addition PadChest is a the largest medical image dataset that has succeeded in applying deep learning based methods for annotation purposes from natural lenguage.  Because of these, the BIMCV-COVID19 dataset http://bimcv.cipf.es/bimcv-projects/bimcv-covid19/ ,which is the largest released dataset on COVID19 pneumonia as of May 20, has applied the same methodology for annotation resulting in 22 differential diagnoses, 122 anatomic locations and 189 different radiological findings mapped onto standard Unified Medical Language System (UMLS) using controlled biomedical vocabulary unique identifiers (CUIs). This code repository contains the annotation pipeline for PadChest and further adapted for the annotation of BIMCV-COVID19 dataset.

## PADCHEST Folder Structure: 
Images are distributed in 54 zip files adding up to 1 TB. 

The file "PADCHEST_chest_x_ray_images_labels_160K.csv" provides the following information for each image:

  - ImageID,
  
  - ImageDir,
  
  - StudyDate_DICOM, 
  
  - StudyID, 
  
  - PatientID, 
  
  - PatientBirth, 
  
  - PatientSex_DICOM, 
  
  - ViewPosition_DICOM, 
  
  - Projection,

  - Pediatric,

  - MethodProjection, 
  
  - Modality_DICOM, 
  
  - Manufacturer_DICOM, 
  
  - PhotometricInterpretation_DICOM, 
  
  - PixelRepresentation_DICOM, 
  
  - PixelAspectRatio_DICOM, 
  
  - SpatialResolution_DICOM, 
  
  - BitsStored_DICOM, 
  
  - WindowCenter_DICOM,
  
  - WindowWidth_DICOM, 
  
  - Rows_DICOM, 
  
  - Columns_DICOM, 
  
  - XRayTubeCurrent_DICOM, 
  
  - Exposure_DICOM, 
  
  - ExposureInuAs_DICOM, 
  
  - ExposureTime_DICOM,
  
  - RelativeXRayExposure_DICOM, 
  
  - ReportID, 
  
  - Report, 
  
  - MethodLabel, 
  
  - Labels, 
  
  - Localizations, 
  
  - LabelsLocalizationsBySentence, 
  
  - LabelCUIS, 
  
  - LocalizationsCUIS
  
There are two types of fields: 

1. Fields with suffix DICOM contains the values of the original field in the DICOM standard [ref 3]. DICOM® (Digital Imaging and Communications in Medicine) is the international standard to transmit, store, retrieve, print, process, and display medical imaging information.  Consult [DICOM standard]( https://www.dicomstandard.org) for field descriptions. 

2. All other non DICOM fields enrich the PADCHEST dataset with additional information.
- Projection: A working classification of the 5 main x-ray projections identified. 
- Report: This field contains the radiological interpretation snippet extracted from the original study report. The text is preprocessed, words are stemmed and tokenized. Each sentence is separated by ‘.’.
- LabelsLocalizationsBySentence: This field contains the anatomic locations for each label as a sequence that follows the order of sentences in a report. Locations are always preceded by the token "loc" so to differentiate them from labels of differential diagnosis and radiological findings. The sequence repeats the pattern formed by one label followed by none or many locations for this label ( label, (0..n) loc name )
- Labels and Localizations fields: Those fields aggregates respectively all different labels and localizations in a report as explained in [1]. 
- LabelCUIS and LocalizationsCUIS: These fields contains the [UMLS Metathesaurus CUIs] (https://uts.nlm.nih.gov/home.html) corresponding to extracted terms.

Examples: 
1. Report: "compar con estudi previ 2010 sin identific cambi signific . siluet cardiomediastin dentr limit normal . no identif imagen condensacion ni opac pulmonar entid signific " 

   Labels: ['unchanged']

   Localizations: ['loc  cardiac']

   LabelsLocalizationsBySentence: [['unchanged'], ['normal', 'loc cardiac'], ['normal']]
   
   LabelsCUIS: []
   
   LocalizationsCUIS: ['C1522601']

2. Report: "imagen pequen taman redond densid metal proyect torax hombre i relacion probabl con perdigon . compresion par lateral izquierd traque probabl estructur vascular . hipoventilacion bibasal ."

   Labels: ['metal', 'superior mediastinal enlargement', 'hypoexpansion basal', 'abnormal foreign body', 'supra aortic elongation']

   Localizations: ['loc  shoulder', 'loc  tracheal, loc  left', 'loc  basal bilateral']

   LabelsLocalizationsBySentence: [['metal', 'abnormal foreign body', 'loc shoulder'], ['superior mediastinal enlargement', 'supra aortic elongation', 'loc tracheal', 'loc left'], ['hypoexpansion basal both sides', 'loc basal bilateral']]
   
   LabelsCUIS: ['C0025552' 'C4273001' '' 'C0016542']
   
   LocalizationsCUIS: ['C0040578' 'C0037004' 'C0443246']

3. Report: "marcapas tricameral . cambi pulmonar cronic . no identific imagen sugest neumotorax . "

   Labels: ['pacemaker', 'chronic changes']

   Localizations: []

   LabelsLocalizationsBySentence: [['pacemaker'], ['chronic changes'], ['normal']]
   
   LabelsCUIS: ['C0030163' 'C0742362']
   
   LocalizationsCUIS: []

The labels are classified in three different treess: differential diagnosis, radiological findings and anatomical locations. Trees and term counts are available in tree_term_CUI_counts_160K.csv.

## Search Instructions for image retrieval by differential diagnosis, Rx findings and anatomical locations: 
To retrieve all relevant images that contains the childs of one term of interest, please consult the appropriate tree and include in the search the corresponding child’s labels or CUIs for this term.
Please note that LabelsLocalizationsBySentence includes all labels for each sentence, and therefore "Normal" or "Exclude" on this field does not imply that the image is "Normal" or should be Excluded (respectively). Instead the labels on this field are the annotations to each sentence and are not aggregated at report level. Therefore, for retrieval and counts of number of images, studies or reports, use the fields Labels, Locations, LabelsCUIs and/or LocalizationsCUIS.  

Ref:

[1] A. Bustos, A. Pertusa, JM. Salinas, M. de la Iglesia. PadChest: A large chest x-ray image dataset with multi-label annotated reports. (Publication Ongoing)

[2] NEMA PS3 / ISO 12052, Digital Imaging and Communications in Medicine (DICOM) Standard, National Electrical Manufacturers Association, Rosslyn, VA, USA


