# Temporal Neighborhood Structure Configuration Model


![build](https://github.com/Feelx234/temp-nestmodel/actions/workflows/pythonapp.yml/badge.svg)

Temporal Neighborhood Structure Configuration Model extends the normal Neighborhood Structure Configuration Models


# Installation instructions
You can install this library like any other python package, depending packages are also installed along with it.

### Installing tnestmodel
Nestmodel is available on pypi thus you can simply install it via:
```
pip install tnestmodel
```

### Installing nestmodel from source
```
git clone https://github.com/Feelx234/nestmodel.git
pip install tnestmodel
```
The installation should take less than a minute.
If you also want to notebooks in the scripts folder please install `jupyter` as outlined below.
```
pip install jupyter
```

### Making sure the installation went smoothly

To make sure the installation has succeeded we need to install pytest as well and then run the tests
```
pip install pytest
python -m pytest tnestmodel
```
If you do not have the graph tools library installed, the tests will emit a warning that it is not installed which can be ignored for now.


# Simple usage of nestmodel
```
pip install tnestmodel
```

```python
from nestmodel.unified_functions import rewire_graph
import networkx as nx
G = nx.karate_club_graph()
G_sample = rewire_graph(G, depth=1).to_nx()
```



