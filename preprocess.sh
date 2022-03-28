wget https://archive.ics.uci.edu/ml/machine-learning-databases/00506/casas-dataset.zip -P ./data/
unzip ./data/casas-dataset.zip
rm ./data/casas-dataset.zip
./mergecsv.sh "./data/*/*.ann.features.csv" ./data/1.0subset.csv
