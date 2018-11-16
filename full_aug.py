from readvid import fill
sets=['D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','none']
for i in sets:
    print(f'Augmenting dataset: {i}')
    fill(i, True)
print('All done!')
