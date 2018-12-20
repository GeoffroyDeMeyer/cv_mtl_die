from collections import OrderedDict

try:
    from name_extraction.name_extractor import NameExtractor    
except ModuleNotFoundError:
    from name_extractor import NameExtractor

MIN_SCALE, MAX_SCALE, STEP_SCALE = 1, 2, 0.05
THA_PSMS = ENG_PSMS = [3, 8, 12]
THA_MODELS = ['--oem 1 --psm {} -c preserve_interword_spaces=1'.format(psm) for psm in THA_PSMS]
ENG_MODELS = ['--oem 1 --psm {}'.format(psm) for psm in ENG_PSMS]

ne = NameExtractor()

def extract_names(imgs_dict,
                    min_scale=MIN_SCALE,
                    max_scale=MAX_SCALE, 
                    step_scale=STEP_SCALE, 
                    tha_models=THA_MODELS, 
                    eng_models=ENG_MODELS):
    '''
    Main name extraction function

    Arguments:
    imgs_dict   -- Dict with the 3 images

    Keywords arguments:
    min_scale -- Minimum image scaling to apply (int)
    max_scale -- Maximum image scaling to apply (int)
    step_scale -- Step between each image scaling (double)
    tha_models -- Tesseract configurations for Thai (list(str))
    eng_models -- Tesseract configurations for English (list(str))
    '''
    if imgs_dict is None:
        return OrderedDict([
            ('tha_title',           ''),
            ('tha_firstname',       ''),
            ('tha_lastname',        ''),
            ('eng_title',           ''),
            ('eng_firstname',       ''),
            ('eng_lastname',        ''),
            ('tha_title_conf',      0),
            ('tha_firstname_conf',  0),
            ('tha_lastname_conf',   0),
            ('eng_title_conf',      0),
            ('eng_firstname_conf',  0),
            ('eng_lastname_conf',   0)
            ])

    return ne.run(imgs_dict['name_tha'], 
                    imgs_dict['name_eng'], 
                    imgs_dict['last_name_eng'], 
                    min_scale, 
                    max_scale, 
                    step_scale, 
                    tha_models,
                    eng_models)