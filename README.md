# Hierarchical cross-entropy loss improves atlas-scale single-cell annotation models

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the code used for "Hierarchical cross-entropy loss improves atlas-scale single-cell annotation models". The paper is available on [bioRxiv](https://doi.org/10.1101/2025.04.22.1234567).

## Repository Information
This repository is partially derived from the [scTab study](https://github.com/theislab/scTab). We have extended and modified the original codebase to implement the hierarchical cross-entropy loss and the experiments described in the paper.

![Fig. 1](images/figure1.png)

## Training Data
The model training uses the CELLxGENE census version "2023-05-15" preprocessed by [scTab](https://github.com/theislab/scTab), which must be downloaded manually from [this link](https://pklab.med.harvard.edu/felix/data/merlin_cxg_2023_05_15_sf-log1p.tar.gz).

## Evaluation Data
For model evaluation, we use the CELLxGENE census version "2023-12-15" as referenced in the paper. This census version is automatically fetched by the code directly from the [CELLxGENE](https://cellxgene.cziscience.com/) portal when needed.

## Hierarchical Cross-Entropy Loss

![Fig. 2](images/figure2.png)

The hierarchical cross-entropy loss leverages inherent hierarchical structures within classification problems to improve model performance. Unlike standard cross-entropy which treats each class independently, this loss function accounts for inclusion relationships between classes. Here we provide a standalone implementation that can be applied to any hierarchical classification task, regardless of the domain or model architecture.

### Reachability Matrix
The function relies on a **reachability matrix** that encodes the hierarchical structure as a directed acyclic graph (DAG). In this matrix:
- Element (i,j) equals 1 if class j is reachable from class i (meaning j is either i itself or j is a subclass of i in the hierarchy)
- Element (i,j) equals 0 otherwise

For example, consider this simple hierarchical structure:
```
    A
   ↙ ↘
  B   C
 ↙ ↘ ↙
D   E
```

The corresponding reachability matrix would be:
```
    A B C D E
A | 1 1 1 1 1
B | 0 1 0 1 1
C | 0 0 1 0 1
D | 0 0 0 1 0
E | 0 0 0 0 1
```

The reachability relation encoded in this matrix is a partial order and has the following mathematical properties:
- **Reflexive**: Every class is reachable from itself (diagonal elements are 1)
- **Antisymmetric**: If class i can reach j and j can reach i, then i equals j
- **Transitive**: If class i can reach j and j can reach k, then i can reach k

### Implementation
```python
def hierarchical_cross_entropy_loss(logits, targets, reachability_matrix, weight=None):
    """
    Hierarchical Cross-Entropy loss
    
    Args:
        logits: Raw model predictions (batch_size, num_classes)
        targets: Ground truth class indices (batch_size)
        reachability_matrix: Matrix encoding hierarchical relationships (num_classes, num_classes)
        weight: Optional class weights
    
    Returns:
        Hierarchical Cross-Entropy loss value
    """
    # Convert logits to probabilities using softmax
    cell_type_probs = torch.softmax(logits, dim=-1)
    
    # Propagate probabilities through the hierarchy using the reachability matrix
    cell_type_probs = torch.matmul(cell_type_probs, reachability_matrix.T)
    
    # Apply log transform (with numerical stability term) for NLL loss calculation
    cell_type_probs = torch.log(
        cell_type_probs + torch.tensor(1e-6, device=cell_type_probs.device)
    )
    
    # Calculate negative log-likelihood loss with optional class weights
    hce_loss = F.nll_loss(cell_type_probs, targets, weight=weight)
    return hce_loss
```

## Contact
For questions or issues, please contact [Davide D'Ascenzo](mailto:davide.dascenzo.work@gmail.com), [Sebastiano Cultrera di Montesano](mailto:scultrer@broadinstitute.org), or [Lorin Crawford](mailto:lcrawford@microsoft.com).

## Citation (BibTeX)
If you use this code or method in your research, please consider citing the following:

```
@article {token,
	author = {Cultrera di Montesano, Sebastiano and D'Ascenzo, Davide and Raghavan, Srivatsan and Amini, Ava P. and Winter, Peter S. and Crawford, Lorin},
	title = {Hierarchical cross-entropy loss improves atlas-scale single-cell annotation models},
	elocation-id = {},
	year = {2025},
	doi = {},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {},
	eprint = {},
	journal = {bioRxiv}
}
```
## License

This project is available under the MIT License.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

