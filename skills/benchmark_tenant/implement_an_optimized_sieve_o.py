"""Implement an optimized Sieve of Eratosthenes to return primes below 1000."""
def solution():
    limit = 1000
    sieve = [True] * limit
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit, i):
                sieve[j] = False
    return [i for i in range(limit) if sieve[i]]

assert len(solution()) == 168 and solution()[-1] == 997
print('BENCHMARK_UNIT_TEST_PASSED')
