from lpkit.label_propagation import label_propagation

neighbors = [[1, 2], [0, 2], [0, 1],
    [4, 5], [3, 5], [3, 4]]

labels, info = label_propagation(neighbors, seed=217, min_sweeps=1, verify_each_sweep=True)
print(info)
print(labels)