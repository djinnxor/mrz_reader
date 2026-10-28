import json
import os
from glob import glob
import cv2

class Template:
    def __init__(self, json_path, img_path):
        self._json_path = json_path
        self._img_path = img_path
        self.img = None
        self.ratio = 1
        self.fields = []
        self.afterprocess = None
        self.document_type = "unknown"
        self.country = "unknown"
        self.__load()

    def __load(self):
        print('json_path' + self._json_path)
        with open(self._json_path) as json_file:
            data = json.load(json_file)
        self.fields = data['fields']
        
        if 'type' in data:
            self.document_type = data['type']
        if 'country' in data:
            self.country = data['country']
            
        if 'afterprocess' in data:
            self.afterprocess = data['afterprocess']

        self.img = self.__resizer(cv2.imread(self._img_path, 0))
        return self.__scaleFields()

    def __resizer(self, img):
        self.ratio = 800/img.shape[0]
        new_height = 800
        new_width = int(img.shape[1] * self.ratio)

        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

    def __scalePts(self, pt):
        return [int(i*self.ratio) for i in pt]

    def __scaleFields(self):

        for k, v in self.fields.items():
            if self.fields[k] is None:
                continue

            tl = self.__scalePts(self.fields[k]['tl'])
            br = self.__scalePts(self.fields[k]['br'])

            self.fields[k]['tl'] = tl
            self.fields[k]['br'] = br



class Templates:
    def __init__(self, root):
        self._root = root
        self.list = []
        self._index = 0
        self._current = 0

    def load(self):
        json_dir = os.path.join(self._root, 'template_*.json')
        for json_path in glob(json_dir):
            img_path = json_path.split('.json')[0] + '.jpg'
            print(json_path, img_path)
            template = Template(json_path, img_path)
            self.list.append(template)
            self._index += 1
        return self

    def __iter__(self):
        self._current = 0
        return self

    def __getitem__(self, index):
        return self.list[index]

    def __len__(self):
        return len(self.list)

    def __next__(self):
        if self._current < self._index:
            temp = self.list[self._current]
            self._current += 1
            return temp
        else:
            raise StopIteration()
