from data_extraction import get_dataset_from_dir, get_tokens_dataset_from_dir
import pickle

methods_source, methods_names, methods_graphs = get_dataset_from_dir(
    "../corpus/r252-corpus-features/")

pickle.dump({'methods_source': methods_source, 'methods_names': methods_names, 'methods_graphs':
    methods_graphs}, open('data/methods_tokens_graphs2.pkl', 'wb'))
