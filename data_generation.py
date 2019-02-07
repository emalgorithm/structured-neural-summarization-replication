from data_extraction import get_dataset_from_dir
import pickle

methods_source, methods_names, method_graphs = get_dataset_from_dir(
    "../corpus/r252-corpus-features/")

pickle.dump({'methods_source': methods_source, 'methods_names': methods_names, 'method_graphs':
    method_graphs}, open('data/methods_data.pkl', 'wb'))
