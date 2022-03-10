import unittest
import pandas as pd
from classifier.data_loader import DataLoader

#python -m unittest tests/test_data_loader.py
class TestDataLoader(unittest.TestCase):

    # Test a non-existing file
    def test_inexistant_file(self):
        test_file = DataLoader(filename='toto.csv', class_id=1)
        self.assertRaises(FileNotFoundError, test_file.load_data())


    # Test empty file
    def test_empty_file(self):
        test_file = DataLoader(filename='tests/empty_file.csv', class_id=1)
        dataframe = test_file.load_data()
        self.assertEquals(dataframe.size, 0)

    # test file with data
    def test_normal_file(self):
        test_file = DataLoader(filename='tests/file_with_data.csv', class_id=1)
        dataframe = test_file.load_data()
        self.assertEquals(dataframe.size, 6) # 2 lines, 3 columns
        self.assertEquals(list(dataframe.columns), ['content', 'class_name', 'class_id'])

        self.assertEquals(dataframe['content'][0], 'a')
        self.assertEquals(dataframe['class_name'][0], 'b')
        self.assertEquals(dataframe['class_id'][0], 1)

        self.assertEquals(dataframe['content'][1], 'c')
        self.assertEquals(dataframe['class_name'][1], 'd')
        self.assertEquals(dataframe['class_id'][1], 1)


if __name__ == '__main__':
    unittest.main()