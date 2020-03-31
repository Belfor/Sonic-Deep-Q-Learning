# Sonic-Deep-Q-Learning

Este repo es la implementación de un agente para mi Trabajo de fin de grado en Ingenería Informática en la UOC. Se basa en el reto propuesto en 2018 por OpenAi donde se debe entrenar a Sonic en diferentes niveles para posteriormente comprobar su comportamiento en un nivel nunca visto.

pagina [retro-contest](/https://contest.openai.com/2018-1/details/)

## Instalacion

Necesitamos seguir algunos pasos de la web de contest

~~~
python -m retro.import.sega_classics

git clone --recursive https://github.com/openai/retro-contest.git

pip install -e "retro-contest/support[docker,rest]" 
~~~

También necesitaremos las BaseLines de open IA

https://github.com/openai/baselines

~~~
git clone https://github.com/openai/baselines.git
cd baselines

pip install -e .
~~~
