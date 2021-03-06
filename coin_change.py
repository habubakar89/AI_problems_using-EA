import random
import numpy as np
import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore')

d = [1,2,3,4,5,6,7,8,'Fitness'] #denominations
N = 50  #total amount
size = 10 #population size

#population generation
population=[]
for i in range(size):
  c = []
  for j in range (len(d)):
    c.append(random.randint(0,N))
  population.append(c)

population = pd.DataFrame(population,columns=d)
population = population.astype({'Fitness':float}) 
for i in range(size):
    fitness=0
    for j in range(len(d)-1):
      fitness+=population[d[j]][i]*d[j]
    fitness = abs(fitness-N)
    population['Fitness'][i] = 1/(1+float(fitness))

pop = population.copy()
pop['Fitness'] = round(pop['Fitness'],4 )
print("Initial population")
print(pop.reset_index(drop=True))

#genetic algorithm
generation = 1

while(True):
  for i in range(size):
    fitness = 0
    for j in range(len(d)-1):
      fitness += population[d[j]][i]*d[j]

    fitness = abs(fitness-N)
    population['Fitness'][i] = 1/(1+float(fitness))

  population = population.sort_values(by=['Fitness'])
  
  parents = population[-6:].reset_index()
  offspring = population[:3].reset_index()

  #crossover
  for i in range(3):
    r = random.randint(0,len(d)-1)
    for j in range(len(d)-1):
      if(j<r):
        offspring[d[j]][i] = parents[d[j]][i]
      else:
        offspring[d[j]][i] = parents[d[j]][6-i-1]
  
  #mutation
  mutation_p=0.75 #mutation probability
  for i in range(3):
    for j in range(len(d)-1):
      p = random.random()
      if(p>mutation_p):
        offspring[d[j]][i] = random.randint(0,N)

  population = population[3:]
  population = population.append(offspring, ignore_index=True)
  population = population.drop(['index'],axis=1)

  if(max(population['Fitness'])==1):
    break

  if(generation%25 == 0):
    print("\nGeneration ",generation)
    pop=population.copy().sort_values(by=['Fitness']).reset_index(drop=True)
    pop['Fitness']=round(pop['Fitness'],4)
    print(pop)
  generation += 1

population=population.sort_values(by=['Fitness']).reset_index(drop=True)
print("\nFinal Population")
print(population)