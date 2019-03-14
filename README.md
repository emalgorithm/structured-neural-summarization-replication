In order to extract the features from the corpus proto files, run:
python data_generation.py

In order to train a model and evaluate a model, run:
python train.py --model_name="lstm_gcn_to_lstm_attention" --device=cuda:0 --print_every=10000 
--device="cuda" --attention=True --graph=True --iterations=500000
All the possible options when running a model can be seen by running:
python train.py --help

A pretrained version of the best performing model (as a state dictionary) can be downloaded at 
https://drive.google.com/file/d/1fm7hGzr-tziNhUMh8duc8s4j5gWW3uKm/view?usp=sharing



