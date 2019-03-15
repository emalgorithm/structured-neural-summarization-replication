from data_processing.data_extraction import get_dataset_from_dir
import pickle

# Generate data
methods_source, methods_names, methods_graphs = get_dataset_from_dir(
    "../corpus/r252-corpus-features/")

# Store data
pickle.dump({'methods_source': methods_source, 'methods_names': methods_names, 'methods_graphs':
    methods_graphs}, open('data/methods_tokens_graphs.pkl', 'wb'))
