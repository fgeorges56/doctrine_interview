# This class is used to load the datasets.

import pandas as pd


class DataLoader:
	def __init__(self, filename, class_id):
		self.filename = filename 
		self.class_id = class_id

	def load_data(self):
		try:
			df = pd.read_csv(self.filename, names=['content', 'class_name'])
			df['class_id'] = self.class_id
			return df
		except FileNotFoundError:
			print('Input file %s does not exist'%self.filename)
