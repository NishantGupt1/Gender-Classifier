from sklearn import tree

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
# List of lists: total 11
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], 
     [181, 85, 43]]

# Y values each classify a list as male or female 
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y) # fit method trains decision tree on dataset
# predicts inputs as 'male' or 'female' based on data collected from previous lists
prediction = clf.predict([[190, 50, 40]])

# CHALLENGE compare their results and print the best one!

print(prediction)