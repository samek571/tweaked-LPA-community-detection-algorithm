## Info
Waaay too big Graph & not enough RAM => near linear time complexity solution.

We use minimal and no dependency asynchronous label propagation, Graph might be undirected TODO rest once i find more time ffs

## Install and venv setup
`python3 -m venv .venv` \
`source .venv/bin/activate` \
`pip install -e .`

## Run Tests
`PYTHONPATH=src pytest -q`

OR 

`source .venv/bin/activate` \
`pip install -e .` \
and then just \
`pytest -q` (thanks to existing `.ini` file)

## Disclaimer
no windows support, dev is doing linux, deal with it.