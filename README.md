# Feasibility_Analysis_Inner_Speech

We classify the EEG signals for the different conditions: Inner speech vs Pronounced speech and Inner speech vs Visualized Condition using the "Inner Speech Dataset" publicly available at https://openneuro.org/datasets/ds003626, and presented https://doi.org/10.1038/s41597-022-01147-2. 

We use the implementation of a Relevance-Based Pruned Extreme Learning Machine, proposed in https://rdcu.be/cyFP3, available at https://github.com/N-Nieto/Relevance_Based_Pruning.

We found that it is possible to distinguish the EEG signals from the different conditions, but we did *not* analyse the potential of classifying the different classes ("up", "down", "right", "left").

The following figure shows the results of the classification for all the available subjects in the Inner speech vs Visualized Condition comparison.

<img src="images/Results_Inner_vs_Vis.png" width="800">

On the other hand, the second figure shows the results of the classification for all the available subjects in the Inner speech vs Pronounced speech comparison.

<img src="images/Results_Inner_vs_Pron.png" width="800">

To ensure reproducibility, all the codes used in this work, along with the environment used, is presented in this repository.

The complete paper can be accessed in the following link: http://50jaiio.sadio.org.ar/pdfs/asai/ASAI-01.pdf

## Citing this work

