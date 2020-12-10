# The ELMo based information retrieval system dashboard

This interactive dashboard demonstrates using ELMo for information retrieval through sentence vectorization and similiarity search. Enter the text you would like to search in the "Text Input" field. Next, enter the question you would like the answer for into the "Question Input" field. Finally, hit the "Search" button, and the app will find the top 5 most matching sentences. Example text and question pre-populates the fields to demonstrate how the application works.

The text search function is powered through Tensorflow 1 using the pre-trained ELMo model without additional training. The app is built using Dash from Plotly and can be deployed as a Flask service. This application is written using Python 3.

## Installation directions 
A Conda environment can be replicated from included env.yml file to include all the dependencies. Run the following command to create the environment:
```
conda env create -f env.yml
``` 

All the package dependences are also included in requirements.txt if you would like to pip install the packages. 

## Run the code
Activating your environment makes the software you installed in that environment available for use. You will see "(dash_elmo)" in front your bash after activation.
```
conda activate dash_elmo
```

Run the code (change path if it is installed else where.)
```
python dash_ELMO.py
```

Dash apps run on localhost by default, and you can only access them on your own machine. Once the code is running, you can access the app at http://127.0.0.1:8050/ using a web browser. 

If you would like to deploy the application on a server that can be accessed, you can follow the instructions at https://dash.plotly.com/deployment


## Author
- Jiangwei Yao

## License 
Apache License 2.0

