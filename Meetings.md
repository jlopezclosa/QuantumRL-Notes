# 23-07-2025

Embedding de la situación que sería buena. Ese embedding sería el que tendría cada uno de los agentes. Podría ser un one hot. Modificar el punto de vista del agente para que siempre sea el primero, pero todos los agentes tienen la misma información en el fondo. Así estás entrenando también para permutaciones. 

Empezar con VDN. La cuestión sería ver si el sistema multiagente puede aprender.
Otros tipos de preprocesamiento de los datos:
Max Pooling cambiar por attention pooling, para no eliminar tanta información (sobre todo en el futuro). 
En vez del max pooling utilizan el transformer. 

Definición de las acciones:
- Swaps. En VDN retornamos un valor Q, por lo tanto lo que haríamos sería que cada agente da un embedding, en el embedding jace un multilayer, y en la salida tienes el qvalue de los dos qubits. VDN es el que distribuye el refuerzo. Tenemos que ver el tema de reseolver conflictos. Igual podemos definir los swaps a partir de la atención -> los valores más altos de atención entre un qubit y otro. 
- Como introducir simetrías? Como mantenerlas?

Leer: boostrap your own latent. 