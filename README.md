# SJ Chest X-Ray Dataset (note: under construction...)

The SJ dataset is a chest x-ray labeled dataset containing 160K high resolution images with their corresponding labeled reports. Its detailed description and labeling methods are described in [1,2].  
This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License and is downloadble from (coming soon). For attribution, please cite as [1,2]. 

## SJ Folder Structure: 
Images are distributed in 50 zip files adding up to 1 TB. 

The file "SJ_chest_x_ray_images_labels_160K.csv" provides the following information for each image:

  - ImageID,
  
  - ImageDir,
  
  - StudyDate_DICOM, 
  
  - StudyID, 
  
  - PatientID, 
  
  - PatientBirth, 
  
  - PatientSex_DICOM, 
  
  - ViewPosition_DICOM, 
  
  - Projection, 
  
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

2. All other non DICOM fields enrich the SJ dataset with additional information.
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

[1] A. Bustos, A. Pertusa, JM. Salinas, M. de la Iglesia. A Chest X-Ray Image Dataset with Multi-label Annotated Reports Covering More than 300 Radiological Findings, Diagnoses and Locations. (Publication Ongoing)

[2] A. Bustos, A. Pertusa. Multi-label annotation of chest x-ray reports in spanish to controlled terminology using neural networks. (Publication Ongoing)

[3] NEMA PS3 / ISO 12052, Digital Imaging and Communications in Medicine (DICOM) Standard, National Electrical Manufacturers Association, Rosslyn, VA, USA

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
