import numpy as np
import scipy.stats as si
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes option price for European call and put options.

    Parameters:
    S : Current stock price
    K : Strike price
    T : Time to expiration in years
    r : Risk-free interest rate
    sigma : Volatility of the stock
    option_type : Type of option ('call' or 'put')

    """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate the option price
    if option_type == "call":
        option_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == "put":
        option_price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price