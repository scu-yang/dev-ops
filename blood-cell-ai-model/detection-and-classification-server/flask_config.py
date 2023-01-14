import json
class Config:
    def __init__(self, filePath):
        self._filePath = filePath
        self._config = self._init_config()

        self.d_pth = self._config["d_pth"]
        self.c_pth = self._config["c_pth"]
        self.image_folder = self._config["image-folder"]

    def _init_config(self):
        with open(self._filePath) as json_file:
            data = json.load(json_file)
            print(data)
            return data