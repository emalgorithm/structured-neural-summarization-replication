## Extracting the Dataset
In order to extract the features from the corpus proto files, run:

`python data_processing/data_generation.py`

In order for the command to be successful, it is necessary to have a directory 
_corpus/r252-corpus-features_ with the protos of the corpus. Optionally, it is possible to 
downloaded the extracted dataset at https://drive.google.com/file/d/14k4AgOVws4_TfPtDGefXzPn3x2Ph083h/view?usp=sharing. After putting the downloaded file
 under the _data/_ directory (which needs to be created), it is possible to train and evaluate the 
 model.

## Running the Models
In order to train a model and evaluate a model, run:

`python training/train.py --model_name="lstm_gcn_to_lstm_attention" 
--print_every=10000 --attention=True --graph=True --iterations=500000`

All the possible options when running a model can be seen by running:

`python train.py --help`

## Pretrained Models
A pretrained version of the best performing model (as a state dictionary) can be downloaded at 
https://drive.google.com/file/d/1fm7hGzr-tziNhUMh8duc8s4j5gWW3uKm/view?usp=sharing

## High-Level Code Structure
- data_processing/: contains the code for extracting, storing, analysing and processing data
    - data_analysis.ipynb: notebook containing analysis of the extracted data
    - data_extraction.py: contains the logic to extract the features data from the proto files of 
    the corpus
    - data_generation.py: file to be called to generate the features data  
    - data_util.py: contains utilities to work with data
    - text_util.py: contains utilities to work with text
- models/: contains all the code for the different models
    - full_model.py: class of the complete methodNaming model
    - gat_encoder.py: class for the Graph Attention Network encoder
    - gcn_encoder.py: class for the Graph Convolutional Network encoder
    - graph_attention_layer.py: class for the Graph Attention Layer used by the Graph Attention 
    Network 
    - graph_convolutional_layer.py: class for the Graph Convolutional Layer used by the Graph 
    Convolutional Network 
    - lstm_decoder.py: class for the LSTM sequence decoder
    - lstm_encoder.py: class for the LSTM sequence encoder
- training.py: contains code to train and evaluate the models
    - evaluation_util.py: contains utilities to compute evaluation metrics
    - train.py: entry-point for training the models
    - train_model.py: contains logic to train the models

