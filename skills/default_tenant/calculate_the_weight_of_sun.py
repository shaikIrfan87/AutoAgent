"""Autonomous induction"""
import math

# Constants
mass_of_earth = 5.972e24  # kg
radius_of_sun = 6.957e8  # meters
gravitational_constant = 6.67430e-11  # m^3 kg^-1 s^-2

# Calculate the weight of the Sun
weight_of_sun = gravitational_constant * mass_of_earth * (radius_of_sun ** 2)

print(f"The weight of the Sun is approximately {weight_of_sun} Newtons.")
