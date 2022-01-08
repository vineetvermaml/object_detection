class ReadClass:
    def __init__(self, classesFile):
        self.classesFile = classesFile

    def readFileasList(self):
        with open(self.classesFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')
        return classNames
