from WMF import WeightedMF
from IFConverter import IFConverter
import numpy as np
import evaluate
import gen_test

converter = IFConverter()
# converter.get_implicit_feedback('./data/Triplets.csv')
# gen_test.gen_fake_R_train('./data/R-full.txt')
# print("Convert successfully!")
with open('./data/R-train.txt', 'r') as f:
    converter.R = [[float(num) for num in line[:-1].split(' ')] for line in f]
    converter.R = np.array(converter.R)
converter.convert()
P, C = converter.P, converter.C
dict_item, dict_user = converter.get_dictionary()

wmf = WeightedMF(P, C, depth=40, early_stopping=False)
clock = evaluate.Clock()
wmf.fit()
clock.stop()
# wmf.save()
# wmf.load()

# Evaluate MAR@k of first n users
k = 20
n_users = 100
predicts = [wmf.get_recommendations(user, k) for user in range(n_users)]
with open('./data/R-full.txt', 'r') as f:
    targets = [[float(num) for num in line[:-1].split(' ')] for line in f]
    targets = [np.argsort(targets[user])[-k:] for user in range(n_users)]
print(evaluate.mark(predicts, targets, k))
