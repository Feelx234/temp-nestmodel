from pathlib import Path
import numpy as np
from numba import njit
from tnestmodel.temp_utils import temp_undir_to_directed



def get_dataset_folder():
    """Returns the location of the dataset folder"""
    path = Path(__file__).parent.absolute()

    if str(path.name) == "scripts":
        path = path.parent
    if str(path.name) == "tnestmodel":
        path = path.parent
    if str(path.name) == "src":
        path = path.parent
    if str(path.name)!="datasets":
        path = path/"datasets"

    assert path.is_dir()
    return path

@njit(cache=True)
def relabel_edges(E):
    """relabels edges such that they start from zero consecutively"""
    nodes_in_order = np.unique(E[:,:2].ravel())
    d = dict()
    for i, node in enumerate(nodes_in_order):
        d[node] = i
    E_out = np.empty_like(E)
    n = 0
    for i,(u,v,t) in enumerate(E):
        new_u = d[u]
        new_v = d[v]
        if new_u!=new_v:
            E_out[n,0] = new_u
            E_out[n,1] = new_v
            E_out[n,2] = t
            n+=1
    return E_out[:n,:], d


class CSVDataset:
    def __init__(self, name, abbr, color, is_directed, seperator=" ", drop_columns=None):
        self.name = name
        self.abbr = abbr
        self.draw_color = color
        self.is_directed = is_directed
        self.seperator = seperator
        self.mapping = None
        self.num_nodes = None
        self.num_edges = None

        self.drop_columns=drop_columns

    def read_pd(self):
        """reads the dataset as pandas DataFrame"""
        import pandas as pd # pylint: disable=import-outside-toplevel
        df = pd.read_csv(get_dataset_folder()/(self.name+".csv"), sep= self.seperator, header=None)
        self.num_edges = len(df)
        return df

    def __str__(self):
        return f"CSVDataset({self.name}, {self.is_directed})"


    def read_edges(self):
        """Returns temporal edges for dataset, edges are potentially (un)directed
        nodes are renamed such that they start from 0 consecutively and nodes with no degree are omitted entirely
        """
        df = self.read_pd()
        if self.drop_columns is not None:
            df = df.drop(self.drop_columns, axis=1)
            assert len(df.columns) == 3
            df.columns=[0,1,2]


        time = df[2]
        order = np.argsort(time)

        time = time[order]
        if "float" in str(time.dtype):
            min_val = np.diff(time).min()
            factor = 10
            while min_val*factor < 1:
                factor*=10
            time *= factor
            time = np.array(np.round(time), dtype=np.int64)
        E = np.empty((len(df), 3), dtype=np.int64)
        E[:,0] = df[0][order]
        E[:,1] = df[1][order]
        E[:,2] = time

        if not self.is_directed:
            tmp_min = np.minimum(E[:,0], E[:,1])
            tmp_max = np.maximum(E[:,0], E[:,1])
            E[:,0]=tmp_min
            E[:,1]=tmp_max
        unique_edges = set(map(tuple, E))
        E = np.array(list(unique_edges), dtype=np.int64)
        order = np.argsort(E[:,2].ravel())
        E= E[order, :]


        E, mapping = relabel_edges(E)
        self.mapping = mapping
        self.num_nodes = len(mapping)
        self.num_edges = len(E)
        return E

    def read_edges_dir(self):
        """Returns directed temporal edges for dataset, undirected datasets have two entries for each edge"""
        E = self.read_edges()
        if not self.is_directed:
            E = temp_undir_to_directed(E)
        return E




datasets = [
    CSVDataset(name="aves-weaver-social",
            abbr="wea",
            color="black",
            is_directed=False,
            seperator = " "),
    CSVDataset(name="ht09",
               abbr="ht09",
               color="red",
               is_directed=False,
               seperator = "\t"),
    CSVDataset(name="workplace_2013",
               abbr="wp",
               color="green",
               is_directed=False,
               seperator = " ",),
    CSVDataset(name="mammalia-primate-association",
               abbr="pri",
               color="lightblue",
               is_directed=False,
               seperator = " ",
               drop_columns=[2]),
    CSVDataset(name="mammalia-raccoon-proximity",
               abbr="rac",
               color="darkblue",
               is_directed=False,
               seperator = " ",
               drop_columns=[2]),

    CSVDataset(name="talk_eo",
               abbr="talk_eo",
               color="royalblue",
               is_directed=True,
               seperator = ","),
    CSVDataset(name="dnc",
               abbr="dnc",
               color="brown",
               is_directed=True,
               seperator = ","),
    CSVDataset(name="fb-forum",
               abbr="fb",
               color="orange",
               is_directed=True,
               seperator = ","),
    CSVDataset(name="email-eu2",
               abbr="eu2",
               color="magenta",
               is_directed=True,
               seperator = " "),
    CSVDataset(name="email-eu3",
               abbr="eu3",
               color="yellow",
               is_directed=True,
               seperator = " "),
]
other_datasets = [
    CSVDataset(name="hospital_ward",
            abbr="hw",
            color="blue",
            is_directed=False,
            seperator = "\t"),
    CSVDataset(name="opsahl",
            abbr="opsahl",
            color="black",
            is_directed=True,
            seperator = ","),
    CSVDataset(name="dblp",
               abbr="dblp",
               color="cyan",
               is_directed="???",
               seperator = ","),
    CSVDataset(name="college",
               abbr="college",
               color="gray",
               is_directed="???",
               seperator = " "),
    CSVDataset(name="soc-bitcoin",
               abbr="bitcoin",
               color="pink",
               is_directed="???",
               seperator = ","),
    CSVDataset(name="talk_cy",
               abbr="talk_cy",
               color="lighsteelblue",
               is_directed="???",
               seperator = ","),
    CSVDataset(name="highschool_2011",
               abbr="hs11",
               color="purple",
               is_directed=False,
               seperator = "\t"),
]
