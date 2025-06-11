def calculate_rps(rce: float, tcr: float) -> float:
    """
    Calculate RPS (Result Percentage) using the formula: RPS = (RCE/TCR) * 100%
    
    Args:
        rce (float): The RCE value (Result/Current)
        tcr (float): The TCR value (Total/Reference)
        
    Returns:
        float: The calculated percentage
        
    Raises:
        ZeroDivisionError: If TCR is 0
    """
    if tcr == 0:
        raise ZeroDivisionError("TCR cannot be zero")
    
    return (rce / tcr) * 100 