import os
import glob
import json
from typing import Callable

class DatasetLoader():
    """
    This class is for loading datasets for machine learning purposes.
    By default, it can be used to index a folder consisting of files with specific extensions.
    Strategies to define different partitions, or how to define targets can be passed on.

    Attributes
    ----------
    files_dir: str
        Path to the directory where the files are stored.
    json_path: str
        Path to a json path indexing the files, and other necessary information.
    items: dict
        The dictionary containing all the items of the dataset.
    process_strategy: function
        The function to define the processing strategy when getting an item.
    train_keys: [str]
        The keys related to training items.
    dev_keys: [str]
        The keys related to development items.
    test_keys: [str]
        The keys related to testing items.

    Methods
    -------
    load_from_json()
        Load items dict from the json file.
    save_to_json()
        Save items dict into the json file.
    index_files(strategy: function(str) -> {str:str})
        Given a strategy index the files in the given directory.

    
    """
    def __init__(self, 
                 files_dir:str, 
                 json_path:str):
        """Get the object of AudioDatasetLoader

        Parameters
        ----------
        files_dir: str
            Path to the files directory.
        json_path: str
            Path to a json path indexing the files, and other necessary information.
        items: dict
            The dictionary containing all the items of the dataset.
       """
        self.files_dir = files_dir
        self.json_path = json_path
        self.process_strategty = lambda x: x
        self.items = {} 
        self.train_keys = []
        self.dev_keys = []
        self.test_keys = []

    def load_from_json(self):
        """Saves the items dictionary to the json file.
        """
        with open(self.json_path, 'r') as json_file: 
            self.items = json.load(json_file)
    
    def save_to_json(self):
        json_dir = os.path.dirname(self.json_path)
        if not os.path.exists(json_dir) and json_dir != "":
            os.makedirs(json_dir)
        with open(self.json_path, 'w') as json_file:
            json.dump(self.items, json_file,  indent=4, ensure_ascii=False)

    def index_files(self, ext:str, strategy: Callable[[str], [str, {str: str}]], **kwargs) -> None:
        """Index files to the items dictionary, given a strategy and an extention of files.
        The strategy gets the file_path and returns (id:str, item:dict) 
        """
        file_paths = os.path.join(self.files_dir, "**", f"*.{ext}")
        file_paths = glob.glob(file_paths, recursive=True)
        for i, file_path in enumerate(file_paths):
            self.print_bar(i + 1, len(file_paths), prefix="Indexing files:", length=40)
            item_id, item_dict = strategy(file_path, **kwargs)
            self.items[item_id] = item_dict
           
    def filter_items(self, filter_key: str, filter_value: str) -> {str: {str: str}}:
        """Filter items based on a specific key in items dictionary.
        """
        new_data = {}
        for item_key, item in self.items.items():
            if item[filter_key] == filter_value:
                new_data[item_key] = item
        return new_data

    @staticmethod
    def print_bar (iteration: int, total: int, prefix = '', suffix = '', decimals = 1, length = "fit", fill = '█') -> None:
        """Prints a progress bar on the terminal

        Example
        -------
        for i in range(50):
            print_bar(i+1, 50, prefix="simple loop:", length=40)
        """
        if length=="fit":
            rows, columns = os.popen('stty size', 'r').read().split() # checks how wide the terminal width is
            length = int(columns) // 2
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        if iteration == total: # go to new line when the progress bar is finished
            print()

    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, key):
        item = self.items[key]
        output = self.process_strategty(item)
        return output


