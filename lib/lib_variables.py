code_to_continent = {
    1 : "AF",
    2 : "EU",
    3 : "AS",
    4 : "AS",
    5 : "OC",
    6 : "SA",
    7 : "NA",
    8 : "NA",
    9 : "NA"
}

continent_to_station = {
    "AF" : ['DWA'],
    "AS" : ['MLIT'],
    "EU" : ['EAU', 'DEFRA'],
    "NA" : ['WSC', 'USGS', 'MEFCCWP'],
    "OC" : ['ABOM'],
    "SA" : ['DGA']
}

station_to_continent={
    'DWA': "AF",
    'EAU': "EU",
    'WSC': "NA",
    'ABOM': "OC",
    'USGS': "NA",
    'DEFRA': "EU",
    'MEFCCWP': "NA",
    'MLIT': "AS",
}

constraint_to_model_name = {
    "constrained": "GRADES",
    "unconstrained": "WBM"
}

equations_dict={}
equations_dict["ManningLW"] = {
        "parameters": ["a0", "n"] #[a0, n]
        }

equations_dict["DarcyW"] = {
        "parameters": ["a0", "cf"] 
        }

equations_dict["ManningVK"] = {
        "parameters": ["a0", "alpha", "beta"] 
        }
