import unittest
import pandas as pd
from classifier.text_cleaner import preprocess

#python -m unittest tests/test_text_cleaner.py
class TestTextCleaner(unittest.TestCase):

    # Test a sentence transformation.
    # Ponctuation, lower/upper case, html tags
    def test_preprocess_sentence(self):
        input_text = """CeCi Est un test. <br/><p>pour voir si le\npreprocessing fonctionne correctement</p>"""
        objective = """ceci test voir si preprocessing fonctionne correctement"""

        input_text_df = pd.DataFrame([input_text], columns=['content'])
        objective_text_df = pd.DataFrame([objective], columns=['content'])

        self.assertEqual(preprocess(input_text_df)['content'][0], objective_text_df['content'][0])


    # Test an empty sentence transformation.
    # Must ensure an empty input makes an empty output
    def test_preprocess_empty(self):
        input_text = ""
        objective = ""

        input_text_df = pd.DataFrame([input_text], columns=['content'])
        objective_text_df = pd.DataFrame([objective], columns=['content'])

        self.assertEqual(preprocess(input_text_df)['content'][0], objective_text_df['content'][0])
        
if __name__ == '__main__':
    unittest.main()