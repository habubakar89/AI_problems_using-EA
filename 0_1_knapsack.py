import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

limit = 40        #size of knapsack
item_count = 10

weight = sorted(np.random.randint(1,30,size=item_count))      #weight of items
profit = np.random.randint(10,100,size=item_count)    #profit for each
item_count = len(weight)

print('Weight\tProfit')
for i in range(item_count):
  print('%s\t%s'%(weight[i],profit[i]))

#fitness function
def calFitness(population,p,w,l):
  for c in range(len(population)):
    chrom = np.array(population.loc[c])[:-1]
    fit = np.dot(chrom,p)
    we = np.dot(chrom,w)
    if(we<=l):
      population.loc[c]['Fitness'] = fit
    else:
      population.loc[c]['Fitness'] = 0

population_size = 12    #number of chromosomes in population
population = np.random.randint(2,size=(population_size,item_count))
population = pd.DataFrame(population,columns=(weight))
population['Fitness'] = np.zeros(population_size).astype('int')
calFitness(population,profit,weight,limit)
population = population.sort_values(by='Fitness').reset_index(drop=True)
print("Initial Population\n", population)


generation = 1   #number of generation
total_generations = 100

while(True):
  parent = population[-4:].reset_index(drop=True)   #select 4 fittest parents
  offspring = parent.copy()

  #one point crossover
  for ind in range(int(len(parent)/2)):
    split = np.random.randint(0,item_count)
    for i in range(split,item_count):
      temp = offspring.loc[ind].iloc[i]
      offspring.loc[ind].iloc[i] = offspring.loc[len(offspring)-1-ind].iloc[i]
      offspring.loc[len(offspring)-1-ind].iloc[i] = temp

  #bit flip mutation
  mutation_probability = 0.5
  for ind in range(len(offspring)):
    for i in range(item_count):
      if(np.random.randn()>mutation_probability):
        offspring.loc[ind].iloc[i] = offspring.loc[ind].iloc[i]^1

  population = population[len(offspring):].append(offspring)
  calFitness(population,profit,weight,limit)
  population = population.sort_values(by='Fitness').reset_index(drop=True)

  if(generation>total_generations):
    break

  generation += 1
  if(generation%10 == 0):
    print("\nGeneration ",generation)
    print(population)

print("\nFinal Population")
print(population)