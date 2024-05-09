import numpy as np

def fisher_z_transform(corr_score):
    """
    Performs Fisher Z-transformation on the input Pearson correlation coefficient. 
    The tranformation is defined as: z = 0.5 * (ln(1 + r) - ln(1 - r)), where r is 
    the Pearson correlation coefficient. 

    Args:
        corr_score (float): Pearson correlation coefficient

    Returns:
        float: Z-transformed Pearson correlation coefficient
    """
    transformed_corr = 0.5 * (np.log1p(corr_score) - np.log1p(-corr_score))
    return transformed_corr