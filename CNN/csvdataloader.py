import csv
import os
from PIL import Image
import numpy as np

class csvdataloader():
    csv_path="F:\\python work\\emoget\\fer2013\\fer2013.csv"
    extract_path="F:\\python work\\emoget\\training"
    train = os.path.join(extract_path, 'train.csv')
    val = os.path.join(extract_path, 'val.csv')
    test = os.path.join(extract_path, 'test.csv')

    def __init__(self):
        pass

    def load_data(self):
        if os.path.isfile(self.train) or os.path.isfile(self.val) or os.path.isfile(self.test):
            print("already exist!")
            #self.load_image_from_csv()
            return None

        with open(self.csv_path) as f:
            csvr = csv.reader(f)
            header = next(csvr)
            rows = [row for row in csvr]

            trn = [row[:-1] for row in rows if row[-1] == 'Training']
            csv.writer(open(self.train, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
            print(len(trn))

            val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
            csv.writer(open(self.val, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
            print(len(val))

            tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
            csv.writer(open(self.test, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
            print(len(tst))
            #self.load_image_from_csv()

    def load_image_from_csv(self):
        train_set = os.path.join(self.extract_path, 'train')
        val_set = os.path.join(self.extract_path, 'val')
        test_set = os.path.join(self.extract_path, 'test')

        for save_path, csv_file in [(train_set, self.train), (val_set, self.val), (test_set, self.test)]:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            with open(csv_file) as f:
                csvr = csv.reader(f)
                header = next(csvr)
                for i, (label, pixel) in enumerate(csvr):
                    pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
                    subfolder = os.path.join(save_path, label)
                    if not os.path.exists(subfolder):
                        os.makedirs(subfolder)
                    im = Image.fromarray(pixel).convert('L')
                    image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
                    im.save(image_name)

