print('\n'.join([''.join([t for v, t in [[3, 'fizz'], [5, 'buzz']]
                          if i % v == 0]) or str(i) for i in range(1, 101)]))
