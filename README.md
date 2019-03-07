# NaiveBayes_HierarchicalClustering
I implemented naive bayes (one of the supervised learning algorithm) and hierarchical clustering (one of unsupervised learning algorithm) with using Python language. I use credit approval [https://archive.ics.uci.edu/ml/datasets/credit+approval] data set for training and testing operation. The data set can be accessed via the given link or the created credit_approval.arff file can be used.

## About Credit Approval Data Set
This data set concerns credit card applications.  All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data. There are many types of attributes in data set like continues (numeric) and nominal with small number of values, and nominal with larger numbers of values. Data set has 690 instances.

![class distribution](https://github.com/metinmertakcay/NaiveBayes_HierarchicalClustering/blob/master/images/class%20distribution.jpg)

Data set has 15 attributes:
* A1 (nominal) {a, b}
* A2 (numeric)
* A3 (numeric)
* A4 (nominal) {u, y, l, t}
* A5 (nominal) {g, p, gg}
* A6 (nominal) {c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff}
* A7 (nominal) {v, h, bb, j, n, z, dd, ff, o}
* A8 (numeric)
* A9 (nominal) {t, f}
* A10 (nominal) {t, f}
* A11 (numeric)
* A12 (nominal) {t, f}
* A13 (nominal) {g, p, s}
* A14 (numeric)
* A15 (numeric)

37 cases have one or more missing values. The missing values from particular attributes are:
* A1: 12
* A2: 12
* A4: 6
* A5: 6
* A6: 9
* A7: 9
* A14: 13

The importance of attributes are ranked as following;

![attribute distinguishing](https://github.com/metinmertakcay/NaiveBayes_HierarchicalClustering/blob/master/images/attribute_distinguishing.jpg)

The most distinguishing attribute is A9, and the least distinguishing attribute is A1. Data set contains 61 outlier sample.

![outlier](https://github.com/metinmertakcay/NaiveBayes_HierarchicalClustering/blob/master/images/outlier.jpg)

### Meaning of Files
Results can be obtained by changing the file names.
* missing_data_removed.txt: Missing datas were filled using Weka.
* missing_outlier_data_removed: Missing datas were filled and outlier datas were removed with using Weka
* missing_outlier_somefeatures_removed: Missing datas were filled, outlier datas and some features were removed with using Weka
