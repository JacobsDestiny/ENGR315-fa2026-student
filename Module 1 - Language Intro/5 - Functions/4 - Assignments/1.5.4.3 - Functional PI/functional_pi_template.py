import math

"""
Implementation of Gauss–Legendre algorithm to approximate PI from https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_algorithm

:param target_error: Desired error for PI estimation
:return: Approximation of PI to specified error bound
"""

### YOUR CODE HERE ###
def my_pi(target_error):
    """
    Implementation of Gauss–Legendre algorithm to approximate PI

    :param target_error: Desired error for PI estimation
    :return: Approximation of PI to specified error bound
    """

    ### YOUR CODE HERE ###
    a = 1.0
    b = 1.0 / math.sqrt(2)
    c = 1.0 / 4.0
    d = 1.0

    pi_estimate = (a + b) ** 2 / (4 * c)

    while abs(math.pi - pi_estimate) > target_error:
        a_next = (a + b) / 2
        b = math.sqrt(a * b)
        c = c - d * (a - a_next) ** 2
        d = 2 * d
        a = a_next

        pi_estimate = (a + b) ** 2 / (4 * c)

    # change this so an actual value is returned
    return pi_estimate
    # change this so an actual value is returned




desired_error = 1E-10

approximation = my_pi(desired_error)

print("Solution returned PI=", approximation)

error = abs(math.pi - approximation)

if error < abs(desired_error):
    print("Solution is acceptable")
else:
    print("Solution is not acceptable")
