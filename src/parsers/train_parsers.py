#!/usr/bin/python

from src.parsers.parser_common import *

class base_train_parser(base_parser):
    def __init__(self, train_config_path):
        self.parser = DefaultConfigParser()
        
        parser = self.parser
        config = {}
        if len(parser.read(train_config_path)) == 0:
            raise ValueError("base_train_parser(): %s not found", train_config_path)
    
        #config["n_epochs"]          = parser.getint("train", "n_epochs", 300)
        config["n_epochs"]          = int(500)
        #config["n_patience"]        = parser.getint("train", "n_patience", 50)
        config["n_patience"]        = int(50)
        #config["bs"]                = parser.getint("train", "bs", 256)
        config["bs"]                = int(256)
        #config["lr"]                = parser.getfloat("train", "lr", 0.001)
        config["lr"]                = float(0.001)
        #config["lr_decay_factor"]   = parser.getfloat("train", "lr_decay_factor", 0.8)
        config["lr_decay_factor"]   = float(0.8)
        #config["l2_weight"]         = parser.getfloat("train", "l2_weight", 0.0001)
        config["l2_weight"]         = float(0.0001)
        #config["max_grad_norm"]     = parser.getfloat("train", "max_grad_norm", None, False)
        config["max_grad_norm"]     = True
        #config["opt"]               = parser.get("train", "opt", "adam")
        config["opt"]               = "adam"
        
        if config["opt"] == "adam":
            config["opt_opts"] = {}
            #config["opt_opts"]["beta1"] = parser.getfloat("opt_opts", "b1", 0.95)
            config["opt_opts"]["beta1"] = float(0.95)
            #config["opt_opts"]["beta2"] = parser.getfloat("opt_opts", "b2", 0.999)
            config["opt_opts"]["beta2"] =float(0.999)
        
        self.config = config

    @staticmethod
    def write_config(config, f):
        f.write("[train]\n")
        for key in ["n_epochs", "n_patience", "bs", "lr", 
                "lr_decay_factor", "l2_weight", "opt"]:
            f.write("%s= %s\n" % (key.ljust(20), str(config[key])))
        if config["max_grad_norm"] is not None:
            f.write("%s= %s\n" % ("max_grad_norm".ljust(20), str(config["max_grad_norm"])))
            
        f.write("\n")
        f.write("[opt_opts]\n")
        if config["opt"] == "adam":
            f.write("%s= %s\n" % ("b1".ljust(20), str(config["opt_opts"]["beta1"])))
            f.write("%s= %s\n" % ("b2".ljust(20), str(config["opt_opts"]["beta2"])))

class vae_train_parser(base_train_parser):
    def __init__(self, train_config_path):
        super(vae_train_parser, self).__init__(train_config_path)
        #self.config["n_steps_per_epoch"] = \
         #       self.parser.getint("train", "n_steps_per_epoch", -1)
        self.config["n_steps_per_epoch"] = int(2000)

    def get_config(self):
        return self.config

    @staticmethod
    def write_config(config, f):
        f.write("[train]\n")
        for key in ["n_steps_per_epoch"]:
            f.write("%s= %s\n" % (key.ljust(20), str(config[key])))
        super(vae_train_parser, vae_train_parser).write_config(config, f)
