import src.concepts as c
from src.neuron.neuron import Neuron

neuron = Neuron()
    
              
rain = c.Rain(1)

#rain not fall down
r1 = [rain.x, rain.not_().x, rain.not_().fall().x ,rain.not_().fall().down().x]

print(r1)
print(all(r1))


do = c.Do(1)

#do you like coffe
r2 = [do.x, do.you().x, do.you().like().x, do.you().like().coffe().x]

print(r2)
print(all(r2))