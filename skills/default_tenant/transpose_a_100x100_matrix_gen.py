"""Autonomous induction"""
def solution():
    m = [[i * 100 + j for j in range(100)] for i in range(100)]
    return [[m[j][i] for j in range(100)] for i in range(100)]
