import yaml

def get_dataset_length(dataset):
    n = 0
    for record in dataset:
        n+=1
    return n

def load_config_file(config_file_path):
    with open(config_file_path,'rb') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    config = convert_parameter(config, 
                               float_params=config['float_params'], 
                               int_params=config['int_params'])
    return config


def convert_parameter(config_dict, float_params, int_params):
    if type(config_dict) != dict:
        #BASECASE 1
        return config_dict
    else:
        for key in config_dict:
            if key == 'params':
                if config_dict[key]:
                    for param in config_dict[key]:
                        if param in float_params:
                            config_dict[key][param] = float(config_dict[key][param])
                        elif param in int_params:
                            config_dict[key][param] = int(config_dict[key][param])
                else:
                    #BASECASE 2
                    config_dict[key] = {}
            else:
                config_dict[key] = convert_parameter(config_dict[key], float_params, int_params)
        return config_dict