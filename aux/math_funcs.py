#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Shayan Gharib, Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = ['avg', 'to_percentage', 'weighting_factors']


def avg(of_this):
    """Returns the average of an iterable object.

    :param of_this: The object.
    :type of_this: list|tuple
    :return: The average value.
    :rtype: float
    """
    return sum(of_this)/float(len(of_this))


def to_percentage(of_this, total_amount):
    """Returns the percentage of the sum of an object, according to total amount. .

    :param of_this: The object.
    :type of_this: list|tuple
    :param total_amount: The total amount.
    :type total_amount: float
    :return: The percentage value.
    :rtype: float
    """
    return 100 * float(sum(of_this)) / total_amount


def weighting_factors(nb_a, nb_b):
    """Returns the weighting sum factors.

    :param nb_a: The amount of quantity a.
    :type nb_a: int
    :param nb_b: The amount of quantity b.
    :type nb_b: int
    :return: The factors for a and b.
    :rtype: (int, int)
    """
    denom = float(nb_a + nb_b)
    return nb_b/denom, nb_a/denom

# EOF
