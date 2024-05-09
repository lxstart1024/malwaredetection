import os
import abstractGraph
import apk2graph
import gml2txt
import numpy
import sys
import Markov as mk
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

def process_graph():
    gmlfile = os.getcwd() + "/gml/"
    txtfile = os.getcwd() + "/graphs/Trial1/"
    num = 0

    for gmlname in os.listdir(gmlfile):
        try:
            storepath = txtfile + gmlname.rpartition(".")[0] + ".txt"
            gmlname = gmlfile + gmlname
            g, edgelist = gml2txt.gml2graph(gmlname)
            gml2txt.caller2callee(edgelist, g.vs, storepath)
        except:
            print(gmlname + "to txt has some ero")
        else:
            print(gmlname + " to txt done")

    logfile = os.getcwd() + "/log.txt"
    with open(logfile, 'w') as log:
        for txtname in os.listdir(txtfile):
            txtpath = txtfile + txtname
            _app_dir = os.getcwd()
            abstractGraph._preprocess_graph(txtpath, _app_dir)
            log.write(txtname.rpartition(".")[0] + ".apk" + " is abstracted" + "\n")
            num += 1
        log.write(str(num) + " apks have done")

def family_to_Mak():
    PACKETS = []
    WHICHCLASS = "Families"
    wf = "Y"
    appslist = None
    dbs = None

    with open("Families11" + '.txt') as packseq:
        for line in packseq:
            PACKETS.append(line.replace('\n', ''))
    packseq.close()
    allnodes = PACKETS
    allnodes.append('self-defined')
    allnodes.append('obfuscated')
    print("allnoedes:", allnodes, "\n")

    Header = []
    Header.append('filename')
    for i in range(0, len(allnodes)):
        for j in range(0, len(allnodes)):
            Header.append(allnodes[i] + 'To' + allnodes[j])
    print('Header is long ', len(Header))

    Fintime = []
    dbcounter = 0

    numApps = os.listdir('family/')

    DatabaseRes = []
    DatabaseRes.append(Header)

    leng = len(numApps)
    for i in range(0, len(numApps)):
        print('starting ', i + 1, ' of ', leng)
        if wf == 'Y':
            with open('family/' + str(numApps[i])) as callseq:
                specificapp = []
                for line in callseq:
                    specificapp.append(line.replace('\n', ''))
            callseq.close()
        else:
            specificapp = []
            for line in dbs[dbcounter][i]:
                specificapp.append(line)

        Startime = time()
        MarkMat = mk.main(specificapp, allnodes, wf)

        MarkRow = []
        if wf == 'Y':
            MarkRow.append(numApps[i])
        else:
            MarkRow.append(appslist[dbcounter][i])
        for i in range(0, len(MarkMat)):
            for j in range(0, len(MarkMat)):
                MarkRow.append(MarkMat[i][j])

        DatabaseRes.append(MarkRow)
        Fintime.append(time() - Startime)
    dbcounter += 1
    f = open('Features/' + WHICHCLASS + '/' + "result" + '.csv', 'w', encoding="utf-8")
    for line in DatabaseRes:
        f.write(str(line) + '\n')
    f.close()

def package_to_Mak():
    PACKETS = []
    WHICHCLASS = "Packages"
    wf = "Y"
    appslist = None
    dbs = None

    with open(WHICHCLASS + '.txt') as packseq:
        for line in packseq:
            PACKETS.append(line.replace('\n', ''))
    packseq.close()
    allnodes = PACKETS
    allnodes.append('self-defined')
    allnodes.append('obfuscated')

    Header = []
    Header.append('filename')
    for i in range(0, len(allnodes)):
        for j in range(0, len(allnodes)):
            Header.append(allnodes[i] + 'To' + allnodes[j])
    print('Header is long ', len(Header))

    Fintime = []
    dbcounter = 0

    numApps = os.listdir('package/')

    DatabaseRes = []
    DatabaseRes.append(Header)

    leng = len(numApps)
    for i in range(0, len(numApps)):
        print('starting ', i + 1, ' of ', leng)
        if wf == 'Y':
            with open('package/' + str(numApps[i])) as callseq:
                specificapp = []
                for line in callseq:
                    specificapp.append(line.replace('\n', ''))
            callseq.close()
        else:
            specificapp = []
            for line in dbs[dbcounter][i]:
                specificapp.append(line)

        Startime = time()
        MarkMat = mk.main(specificapp, allnodes, wf)

        MarkRow = []
        if wf == 'Y':
            MarkRow.append(numApps[i])
        else:
            MarkRow.append(appslist[dbcounter][i])
        for i in range(0, len(MarkMat)):
            for j in range(0, len(MarkMat)):
                MarkRow.append(MarkMat[i][j])

        DatabaseRes.append(MarkRow)
        Fintime.append(time() - Startime)
    dbcounter += 1
    f = open('Features/' + WHICHCLASS + '/' + "result" + '.csv', 'w', encoding="utf-8")
    for line in DatabaseRes:
        f.write(str(line) + '\n')
    f.close()

def random_forest_family_feature():
    data = pd.read_csv("MaMadroid_family_feature_with_label.csv")

    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    random_forest = RandomForestClassifier(n_estimators=51, max_depth=8)

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1': make_scorer(f1_score)}

    cv_results = cross_validate(random_forest, X, y, cv=10, scoring=scoring)

    print("Cross Validation Results:")
    print("Accuracy:", cv_results['test_accuracy'].mean())
    print("Precision:", cv_results['test_precision'].mean())
    print("Recall:", cv_results['test_recall'].mean())
    print("F1:", cv_results['test_f1'].mean())

def random_forest_package_feature():
    data = pd.read_csv("new_MaMadroid_package_feature_with_label.csv")

    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    random_forest = RandomForestClassifier(n_estimators=101, max_depth=64)

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1': make_scorer(f1_score)}

    cv_results = cross_validate(random_forest, X, y, cv=10, scoring=scoring)

    print("Cross Validation Results:")
    print("Accuracy:", cv_results['test_accuracy'].mean())
    print("Precision:", cv_results['test_precision'].mean())
    print("Recall:", cv_results['test_recall'].mean())
    print("F1:", cv_results['test_f1'].mean())