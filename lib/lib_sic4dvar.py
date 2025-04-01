from netCDF4 import Dataset


class Sic4dvarResults(object):
    def __init__(self, sic4dvar_path):
        self.sic4dvar = Dataset(sic4dvar_path)
    
    def close(self):
        self.sic4dvar.close()


class Sic4dvarResultsMM(Sic4dvarResults):
    def __init__(self, sic4dvar_path):
        super().__init__(sic4dvar_path)
    
    def get_time(self):
        pass


class Sic4dvarResultsConfluence(Sic4dvarResults):
    def __init__(self, sic4dvar_path):
        super().__init__(sic4dvar_path)
    
    def get_time(self):
        pass
        