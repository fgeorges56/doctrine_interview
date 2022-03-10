from data_loader import DataLoader
from text_cleaner import preprocess
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split


# we have several dataframes.
# They may be imbalanced. We address this issue.
def merge_datasets(dataframe_array, max_nb_records=200000):
	biggest_dataframe_size = min(
									max([x.shape[0] for x in dataframe_array]),
									max_nb_records) # I do not want too many records, to avoid memory problems
	#print("biggest_dataframe_size", biggest_dataframe_size)
	res = pd.DataFrame()
	for d in dataframe_array:
		if d.shape[0] < biggest_dataframe_size:
			res = pd.concat([res, resample(d, 
								replace=True, # sample with replacement
								n_samples=biggest_dataframe_size) ])
		else:
			res = pd.concat([res, d.sample(n=biggest_dataframe_size) ])
	return res



# First, we load the datasets.
print("Loading CRIM dataset")
crim_data_loader = DataLoader(filename='../data/CRIM.csv', class_id=1)
crim_data = crim_data_loader.load_data() # 385 lines

print("Loading COM dataset")
com_data_loader = DataLoader(filename='../data/COM.csv', class_id=2)
com_data = com_data_loader.load_data() # 2263 lines

print("Loading CIV dataset")
civ_data_loader = DataLoader(filename='../data/CIV.csv', class_id=3)
civ_data = civ_data_loader.load_data() # 13085 lines

print("Loading SOC dataset")
soc_data_loader = DataLoader(filename='../data/SOC.csv', class_id=4)
soc_data = soc_data_loader.load_data() # 12260 lines



# Second, whe aggregate the datasets.
# We also make sure the data is not imbalanced, by resampling least frequent classes
print("Aggregating datasets")
full_dataframe = merge_datasets([crim_data, com_data, civ_data, soc_data])


# Third, we clean the content
print("Cleaning text")
full_data_cleaned = preprocess(full_dataframe) 


# Fourth, we build the classifier

# Term-Frequency Inversed Document Frequency
# then Singular-Value Decomposition
# Used to keep most relevant tokens
vec = TfidfVectorizer()
x = vec.fit_transform(full_data_cleaned["content"])


# This parameter has been decided empirically. It has the value that maximizes both teh accuracy and recall of each class.
# Without overfitting, of course.
n_components = 150

svd = TruncatedSVD(n_components=n_components) # This parameter has to be adapted, to best fit our situation.
fitted_x = svd.fit_transform(x)

y = full_data_cleaned["class_id"].values
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')

metrics = cross_validate(model, fitted_x, y, scoring=['precision_macro', 'recall_macro'], cv=cv, n_jobs=-1)
print('Precision: %.3f (%.3f)' % (mean(metrics["test_precision_macro"]), std(metrics["test_precision_macro"])))
print('Recall: %.3f (%.3f)' % (mean(metrics["test_recall_macro"]), std(metrics["test_recall_macro"])))


	

# I also want a confusion matrix, to see which classes are less fitted to the model.
# To test a confusion matrix, I split in 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(fitted_x, y, test_size=0.2)

fitted_model = model.fit(X_train, y_train)
pred_y = model.predict(X_test)
print(confusion_matrix(y_test, pred_y))

