#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import pywt
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split,StratifiedKFold,cross_val_score
    from sklearn.learning_curve import learning_curve,validation_curve
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,learning_curve,validation_curve,GridSearchCV
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.externals import joblib
from joblib import Parallel,delayed
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer,mean_squared_error,confusion_matrix,accuracy_score
from xgboost import XGBClassifier,plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
sampleRate = 1000;eachSampleDuration = 5;constantLength = sampleRate * eachSampleDuration
gmlConstant = 1000#really use legth
dataFileName = ['./powderNone.csv', './powderTest.csv', './narcotNone.csv', './narcotTest.csv']
testName = dataFileName[3];noneName = dataFileName[2]
useAllData = True#acquire much more data is difficult
quickRun = False#;useMatlabFilteredData = True
powder_none_test_num_lines = sum(1 for line in open(noneName))
powder_test_num_lines = sum(1 for line in open(testName))
total_sample_count = (powder_none_test_num_lines + powder_test_num_lines) // constantLength
if useAllData:
    total_sample_count = total_sample_count * 3
total_set = [[.0 for x in range(constantLength + 1)] for y in range(total_sample_count)]
def readFileAndFilter(fileName):
    with open(fileName, 'r') as f:
	    channel = []
	    for line in f.readlines(): channel.append([float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2])])
    cm = np.matrix(channel)
    del channel[:];del channel
    s2 = np.ravel(cm[:,1])
    if not useAllData:
        data = np.array([s2])
    else:
        data = np.array([np.concatenate((np.ravel(cm[:,0]), s2, np.ravel(cm[:,2])))])
    '''ch_types = ['eeg'];ch_names = ['CZ']
    info = mne.create_info(ch_names=ch_names, sfreq=sampleRate, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    filtereds2 = raw.copy().filter(1., 40.).get_data()[0]
    filtereds2 = mne.filter.filter_data(s2, 1000, 1., 40., picks=None, filter_length='auto', 
	l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs='cuda', method='fir', 
	iir_params=None, copy=True, phase='zero', fir_window='hamming', 
	fir_design='firwin', pad='reflect_limited', verbose=None)'''
    (cA, cD) = pywt.dwt(data, 'db1')
    '''wavelet_names = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7',
                 'db8', 'db9', 'db10', 'sym10', 'coif1', 'coif2',
                 'coif3', 'coif4', 'coif5']'''
    discreteWaveletTransformeds2 = cA.tolist() + cD.tolist()
    #print(len(discreteWaveletTransformeds2))
    return discreteWaveletTransformeds2[0]+discreteWaveletTransformeds2[1]
positiveData = readFileAndFilter(testName)
def printPrecisionRecall(clf, y_test, y_pred):
    print('%s Precision: %.3f' % (clf, precision_score(y_true=y_test, y_pred=y_pred)))
    print('%s Recall: %.3f' % (clf, recall_score(y_true=y_test, y_pred=y_pred)))
    print('%s F1: %.3f' % (clf, f1_score(y_true=y_test, y_pred=y_pred)))
positiveSampleCount = powder_test_num_lines // constantLength
if useAllData:
    positiveSampleCount = positiveSampleCount * 3

for i in range(0, positiveSampleCount):
	for j in range(0, constantLength):         total_set[i][j] = positiveData[constantLength*i+j]		
	total_set[i][constantLength] = 1#;print('positive ApEn:{}'.format(ApEn(np.array(total_set[i][0:5000]), 2, 3)))
negativeData = readFileAndFilter(noneName)
negativeSampleCount = powder_none_test_num_lines // constantLength
if useAllData:
    negativeSampleCount = negativeSampleCount * 3
for i in range(0, negativeSampleCount):
	for j in range(0, constantLength):
		total_set[positiveSampleCount + i][j] = negativeData[constantLength*i+j]
	total_set[positiveSampleCount + i][constantLength] = -1#;U = np.array(total_set[positiveSampleCount + i][0:5000]);print('negative ApEn:{}'.format(ApEn(U, 2, 3)))
total_data_matrix = np.matrix(total_set)
for sampleLength in range(1000, constantLength + 1, 1000):
    for sampleInitialPoint in range(0, constantLength - sampleLength + 1,1000):
        print('now is sampleLength={}, sampleInitialPoint={}'.format(sampleLength, sampleInitialPoint))
        X_set,y = total_data_matrix[:, range(sampleInitialPoint, sampleInitialPoint + sampleLength)],total_data_matrix[:, constantLength]
#X_set,y = total_data_matrix[:,range(0, gmlConstant)],total_data_matrix[:, constantLength]# numpy.matrix access multi column
        X_train,X_test,y_train,y_test=train_test_split(X_set,np.ravel(y),test_size=0.3,random_state=42)
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)
        if not quickRun:
            cov_mat = np.cov(X_train_std.T)
            eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
            #print('\nEigenvalues \n%s' %eigen_vals)
            tot = sum(eigen_vals)
            var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
            cum_var_exp = np.cumsum(var_exp)
            #print('+'*80);print(var_exp);
            print('-'*80);print('biggest eigen value:{},{},{}'.format(cum_var_exp[0], cum_var_exp[1], cum_var_exp[2]))
        '''plt.bar(range(1,5000),var_exp,alpha=.5,align='center',label='individual explained variance')
plt.step(range(1,5000),cum_var_exp,where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()'''
        print('*'*80)
        '''glearning_rate = [0.045, 0.044, 0.043, 0.042, 0.041,0.03,0.02,1e-2,1e-3,1e-4]
clearning_rate = [1e-4,1e-3,1e-2,0.1,1,6.2,6.25, 6.3,6.35, 6.4,6.45, 6.5,6.55,10,1e2,1e3,1e4]
for gamma in glearning_rate:
	for C in clearning_rate:
			svm = SVC(kernel='rbf', random_state=None, gamma=gamma, C=C)
			svm.fit(X_train_std, y_train)
			y_pred = svm.predict(X_test_std)
			if 1 - (y_test != y_pred).sum() / len(y_test) > 0.513:#0.646:
				#print('svm kernel:rbf\tgamma:{}\tC:{}\taccuracy: {}'.format(gamma, C, 1 - (y_test != y_pred).sum() / len(y_test)))
				y_train_pred = svm.predict(X_train_std)
				
				confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
				print(confmat)
				svm_train = accuracy_score(y_train, y_train_pred) 
				svm_test = accuracy_score(y_test, y_pred) 
				print('rbf svm gamma %.3f C %.2f train/test accuracies %.3f/%.3f' % (gamma, C,svm_train, svm_test))
				scores = cross_val_score(estimator=svm, X=X_train_std, y=y_train,cv=10,scoring='roc_auc')
				print('ROC AUC: %0.2f (± %0.2f) svm' % (scores.mean(), scores.std()))'''
        pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4,1e5]
        param_grid = [{'clf__C': param_range, 
               'clf__kernel': ['linear']},
                 {'clf__C': param_range, 
                  'clf__gamma': param_range, 
                  'clf__kernel': ['rbf']}]
        svmModelName = 'GridSearchCVSVM_' + testName[2:8];svmModelName = svmModelName + '.model'
        if not quickRun:
            gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
            gs = gs.fit(X_train, y_train)
            joblib.dump(gs, svmModelName)
        else:
            gs=joblib.load(svmModelName)
        print('best score of svm:{}'.format(gs.best_score_))
        print(gs.best_params_);clf = gs.best_estimator_
        y_pred = clf.predict(X_test)
        printPrecisionRecall('svm', y_test, y_pred)
        bestSVMName = 'bestSVM_' + testName[2:8] + '.model'
        if not quickRun:
            joblib.dump(clf, bestSVMName)
        else:
            clf = joblib.load(bestSVMName)
        clf.fit(X_train, y_train)
        print('svm Test accuracy: %.3f' % clf.score(X_test, y_test))
        svmSearchCV2CrossValCV5Name = 'gridsearchCV2CrossvalCV5_' + testName[2:8] + '.model'
        if not quickRun:
            gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
				  cv=2,
				  n_jobs=-1)
            joblib.dump(gs, svmSearchCV2CrossValCV5Name)
        else:
            gs = joblib.load(svmSearchCV2CrossValCV5Name)
        scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5,n_jobs=-1)
        print('svm CV accuracy: %.3f ± %.3f' % (np.mean(scores), np.std(scores)))
        clf1 = LogisticRegression(penalty='l2',C=0.001,random_state=0)
        clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
        clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
        pipe1= Pipeline([['sc', StandardScaler()], ['clf', clf1]])
        pipe2= Pipeline([['sc', StandardScaler()], ['clf', clf2]])
        pipe3= Pipeline([['sc', StandardScaler()], ['clf', clf3]])
        clf_labels = ['LR', 'DecisionTree', 'KNN']
        print('10-fold cross validation:\n')
        for clf, label in zip([pipe1, pipe2, pipe3], clf_labels):
	        scores = cross_val_score(estimator=clf, X=X_train_std, y=y_train,cv=10,scoring='roc_auc', n_jobs=-1)
	        print('ROC AUC: %0.2f (± %0.2f) [%s]' % (scores.mean(), scores.std(), label))
        gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2,
				  n_jobs=-1)
        scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1)
        print('decision tree CV accuracy: %.3f ± %.3f' % (np.mean(scores), np.std(scores)))
        tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=1,
                              random_state=0)
        ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500, 
                         learning_rate=0.1,
                         random_state=0)
        tree = tree.fit(X_train_std, y_train)
        y_train_pred = tree.predict(X_train_std)
        y_test_pred = tree.predict(X_test_std)
        tree_train = accuracy_score(y_train, y_train_pred)
        tree_test = accuracy_score(y_test, y_test_pred)
        print('base classifier of adaboost Decision tree train/test accuracies %.3f/%.3f'
            % (tree_train, tree_test))
        ada = ada.fit(X_train_std, y_train)
        y_train_pred = ada.predict(X_train_std)
        y_test_pred = ada.predict(X_test_std)
        ada_train = accuracy_score(y_train, y_train_pred) 
        ada_test = accuracy_score(y_test, y_test_pred) 
        print('AdaBoost train/test accuracies %.3f/%.3f'
            % (ada_train, ada_test))
        clf4 = RandomForestClassifier(n_estimators=35, max_depth=None,
            min_samples_split=2, n_jobs=-1,random_state=0)
        scores = cross_val_score(clf4, X_train_std, y_train, n_jobs=-1)                     
        print('ROC AUC: %0.2f (± %0.2f) [RandomForest]' % (scores.mean(), scores.std()))
        clf5 = ExtraTreesClassifier(n_estimators=10, max_depth=None,
            min_samples_split=2, random_state=0)
        scores = cross_val_score(clf5, X_train_std, y_train, n_jobs=-1)
        print('ROC AUC: %0.2f (± %0.2f) [ExtraTrees]' % (scores.mean(), scores.std()))
        pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])
        pipe_lr.fit(X_train_std, y_train)
        print('pca logistic Test Accuracy: %.3f' % pipe_lr.score(X_test_std, y_test))
        y_pred = pipe_lr.predict(X_test_std)
        scores = []
        if Version(sklearn_version) < '0.18':
            kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
        else:
            kfold = StratifiedKFold(n_splits=10,random_state=1).split(X_train_std, y_train)
        for k, (train, test) in enumerate(kfold):
            pipe_lr.fit(X_train_std[train], y_train[train])
            score = pipe_lr.score(X_train_std[test], y_train[test])
            scores.append(score)
            print('pca lr Fold: %s, Acc: %.3f' % (k+1,  score))
	#print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
            print('pca lr Fold: %s, Acc: %.3f' % (k+1, score))
        print('pca lr CV accuracy: %.3f ± %.3f' % (np.mean(scores), np.std(scores)))
        scores = cross_val_score(estimator=pipe_lr,
                         X=X_train_std,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
        print('pca lr CV accuracy scores: %s' % scores)
        print('pca lr CV accuracy: %.3f ± %.3f' % (np.mean(scores), np.std(scores)))
        clf4=RandomForestClassifier()
        pipe_lr = Pipeline([('scl', StandardScaler()),
                    #('clf', LogisticRegression(penalty='l2', random_state=0))]) 
					('clf', SVC(kernel='rbf', random_state=None, gamma=0.1))])
                    #('clf', clf4)])
        train_sizes, train_scores, test_scores =\
                learning_curve(estimator=pipe_lr,
                               X=X_train_std,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=-1)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.plot(train_sizes, train_mean,
            color='blue', marker='o',
            markersize=5, label='training accuracy')
        plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean,
            color='green', linestyle='--',
            marker='s', markersize=5,
            label='validation accuracy')
        plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
        plt.grid();plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy');plt.legend(loc='lower right')
        plt.ylim([0.4, 1.0]);plt.tight_layout()#;plt.show()
        #param_range = np.arange(1, 50)
        train_scores, test_scores = validation_curve(estimator=pipe_lr, 
                                             X=X_train_std, 
                                             y=y_train, 
                                             param_name='clf__C', # 'n_estimators'
                                             param_range=param_range,
                                             scoring='accuracy',
                                             cv=10,
				                             n_jobs=-1)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.plot(param_range, train_mean, 
            color='blue', marker='o', 
            markersize=5, label='training accuracy')
        plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')
        plt.plot(param_range, test_mean, 
            color='green', linestyle='--', 
            marker='s', markersize=5, 
            label='validation accuracy')
        plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')
        plt.grid();plt.title("Validation Curve With svm")
        plt.xscale('log');plt.legend(loc='best')
        plt.xlabel('range of value C');plt.ylabel('Accuracy')
        plt.ylim([0.4, 1.0]);plt.tight_layout()#;plt.show()
        
        scorer = make_scorer(f1_score, pos_label=1)#pos_label=0)
        c_gamma_range = [0.01, 0.1, 1.0, 10.0]
        param_grid = [{'clf__C': c_gamma_range,
               'clf__kernel': ['linear']},
              {'clf__C': c_gamma_range,
               'clf__gamma': c_gamma_range,
               'clf__kernel': ['rbf']}]
        gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
        gs = gs.fit(X_train, y_train)
        print('best score of svm is:{}'.format(gs.best_score_))
        print(gs.best_params_)
        ''''feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        color='lightblue', 
        align='center')
plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()
if Version(sklearn_version) < '0.18':
    X_selected = forest.transform(X_train_std, threshold=0.15)
else:
    from sklearn.feature_selection import SelectFromModel
    sfm = SelectFromModel(forest, threshold=0.15, prefit=True)
    X_selected = sfm.transform(X_train_std)
print(X_selected.shape)'''
        model = XGBClassifier()
        model.fit(X_train_std, y_train)
        '''model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)'''
        y_pred = model.predict(X_test_std)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("xgboost Accuracy: %.2f%%" % (accuracy * 100.0))
        plot_importance(model)#;plt.show()
        learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        param_grid = dict(learning_rate=learning_rate)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
        grid_result = grid_search.fit(X_train_std, y_train)
        print("xgboost Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("xgboost %f (±%f) with: %r" % (mean, stdev, param))
        gbm = lgb.LGBMRegressor(objective='binary',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_jobs=-1)
        gbm.fit(X_train_std, y_train,
            eval_set=[(X_test_std, y_test)],
            eval_metric='l1',
            early_stopping_rounds=5)
        print('lightGBM Start predicting...')
        y_pred = gbm.predict(X_test_std, num_iteration=gbm.best_iteration_)
        print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
        scores = cross_val_score(estimator=gbm, X=X_train_std, y=y_train,cv=10,scoring='roc_auc', n_jobs=-1)
        print('ROC AUC: %0.2f (± %0.2f) [LightGBM]' % (scores.mean(), scores.std()))
        print('Feature importances:', list(gbm.feature_importances_))
        
        lgmModelName = 'LightGBM_' + testName[2:8]+str(sampleLength)+'_'+str(sampleInitialPoint) + '.model'
        if not quickRun:
            estimator = lgb.LGBMRegressor(num_leaves=31)
            param_grid = {    'learning_rate': [0.01, 0.1, 1],
                   'n_estimators': [20, 40]}
            gbm = GridSearchCV(estimator, param_grid, n_jobs=-1)
            gbm.fit(X_train_std, y_train)
            print('lightGBM Best parameters found by grid search are:', gbm.best_params_)
            joblib.dump(gbm, lgmModelName)
        else:
            gbm = joblib.load(lgmModelName)
        scores = cross_val_score(estimator=gbm, X=X_train_std, y=y_train,cv=10,scoring='roc_auc', n_jobs=-1)
        print('ROC AUC: %0.2f (± %0.2f) [LightGBM]' % (scores.mean(), scores.std()))
        printPrecisionRecall('LightGBM', y_test, y_pred.round())
        clf_ldaShrinkage = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X_train_std, y_train)
        clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train_std, y_train)
        y_pred_ldas = clf_ldaShrinkage.predict(X_test_std)
        y_pred_lda = clf_lda.predict(X_test_std)
        predictions = [round(value) for value in y_pred_ldas]
        accuracy = accuracy_score(y_test, predictions)
        print("shrink lda Accuracy: %.2f%%" % (accuracy * 100.0))
        predictions = [round(value) for value in y_pred_lda]
        accuracy = accuracy_score(y_test, predictions)
        print(" lda Accuracy: %.2f%%" % (accuracy * 100.0))
'''import tensorflow as tf 
from keras.models import Sequential
model = Sequential()
from keras.layers import Dense
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=.01,momentum=.9,nesterov=True))
model.fit(x_train, y_train, epochs=5,batch_size=32)
model.train_on_batch(x_batch,y_batch)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes=model.predict(x_test,batch_size=128)
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential([Dense(32,input_shape=(784,)),
				Activation('relu'),
				Dense(10),
				Activation('softmax'),])
model = Sequential()
model.add(Dense(32,input_dim=784))
model.add(Activation('relu'))
#http://blog.csdn.net/ouening/article/details/71079535	
#use soft margin svm have a try
#use one dimension cnn to have a try, use different size of kernel and stride
#use semisupervise unsupervise learning ,reafinforce learning such as q learning try
import keras
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
x_train = np.random.random((5000,3))
y_train = keras.utils.to_categorical(np.random.randint(2, size=(1000,1)), num_classes=2)
x_test = np.random.random((100,3))
y_test = keras.utils.to_categorical(np.random,randint(2, size=(100,1)),num_classes=2)
model=Sequential()
model.add(Dense(64,activation='relu',input_dim=3))
model.add(Dropout(.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(2,activation='softmax'))
sgd = SGD(lr=.01,decay=1e-6,momentum=.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.fit(x_train, y_train,epochs=20, batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
from keras.layers import Input, Embedding,LSTM, Dense
from keras.models import Model 
main_input = Input(shape=(100,), dtype = 'int32', name='main_input')
x = Embedding(output_dim = 512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)
auxiliary_output = Dense(1,activation='sigmoid', name='aux_output')(lstm_out)
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate(lstm_out,auxiliary_input)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1,activation='sigmoid', name='main_output')(x)
model = Model(inputs=[main_input,auxiliary_input], outputs=[main_output,auxiliary_output])
mode.compile(optimizer='rmsprop',loss='binary_crossentropy',loss_weights=[1.,.2])
model.fit([headline_data,additional_data],[labels, labels],epochs=50,batch_size=32)
model.compile(optimizer='rmsprop',loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
		loss_weights={'main_output': 1., 'aux_output': .2})
model.fit({'main_input':headline_data,'aux_input': additional_data},
		{'main_output':labels,'aux_output':labels},
		epochs=50,batch_size=32)
config = model.get_config()
model = Model.from_config(config)
model = Sequential.from_config(config)
from keras.models import model_from_json
json_string  = model.to_json()
model = model_from_json(json_string)
from keras.models import model_from_yaml
yaml_string = model.to_yaml();model = model_from_yaml(yaml_string)'''
