import os

class DataChecker:
    def getDataFolder():
        data_found = False
        for (directory, folders, _) in os.walk("."):
            if "data" in folders: 
                data_found = True
                DataChecker.checkFiles(directory + "/data")
                return directory + "/data"
                
        if data_found == False : 
            raise Exception("Le dossier 'data' n'a pas été trouvé")


    def checkFiles(data_loc):
        current_files = [(f.name,os.path.isdir(f)) for f in os.scandir(data_loc)]
        needed_files = DataNeededFiles.getNeededFiles()
        for checker in needed_files:
            if not checker in current_files:
                f_precision = "dossier" if checker[1] else "fichier"
                raise Exception(f"Le {f_precision} {checker[0]} est manquant dans le dossier 'data'")
      
class DataNeededFiles:
    def getNeededFiles():
        return [('reduced_images', True),
                ('X_test_BEhvxAN.csv', False),
                ('X_train_J01Z4CN.csv', False),
                ('y_train_OXxrJt1.csv', False)]

    
