# GaussMix


Dependencies: 
- Numpy
- ScikitLearn
- Python 3
- Pandas
- Scipy

To run the program, run it as python(or python3) in your command line.

Arguments:
 -f -file   followed by pathname to dataset
 -h --help   for help
 -x --extra  to get extra plots
 -s -- save  to save the plot as a file in a directory

Examples:

python clusteringviz.py -h
  - To get help using the program
 
python clusteringviz -f path/to/seeds.txt
  - To run the program on a dataset, and draw a scatter plot of both the KMeans and Gaussian algorhitms.

python clusteringviz -f path/to/seeds.txt -x
	- Display extra plots of features

python clusteringviz -f path/to/seeds.txt -s path/to/directory/picture.png
	- saves the plot in a directory, in this case as a png



