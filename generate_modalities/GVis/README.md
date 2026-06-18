## Environment Setup
First make an environment with **python 3.11**

```
mamba create --name CADVisualization python==3.11
```

Then install pip requirements:

```
mamba activate CADVisualization
pip install -r requirements.txt
```

Then install `pythonocc-core` using conda:

```
mamba install pythonocc-core
```

Then update numpy:

```
pip install -U numpy
```

Then to get blender to work with OCC install a lower version of freeimage binaries with ignored dependancies:

```
mamba install freeimage==3.17 --no-deps
```

Now you can run the demo notebook in the main folder.
