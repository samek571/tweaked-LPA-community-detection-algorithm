## Info
Waaay too big Graph & not enough RAM => near linear time complexity solution.

We use minimal and no dependency asynchronous label propagation, Graph might be undirected TODO rest once i find more time ffs

## Install and venv setup
`python3 -m venv .venv` \
`source .venv/bin/activate`
- if `python3 -m venv gives an error`, install it: `sudo apt install python3-venv`

`pip install -e .` (registers lpkit)

## Run Pre-Made Tests
`PYTHONPATH=src pytest -q` or simply `pytest -q` (thanks to existing `.ini` file)

OR run your own using cli

```sh
lpkit ram --in examples/simple.edgelist \
          --out simple.labels.npy \
          --seed 1337 \
          --max-sweeps 200
```

```sh
lpkit stream --in path/to/big_graph.edgelist \
             --out labels.npy \
             --seed 1337 \
             --max-sweeps 50 \
             --block-size 5000
```

## CLI help
`lpkit --help`\
`lpkit ram --help`\
`lpkit stream --help`

## Disclaimer
no windows support, dev is doing linux. Some things might now work such as external sort.