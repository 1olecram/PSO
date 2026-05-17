import numpy as np

class Particle:
    """
    Classe que representa uma partícula no enxame do PSO.
    """
    def __init__(self, n, bounds):
        # Posição inicial aleatória dentro dos limites
        self.position = np.random.uniform(bounds[0], bounds[1], n)
        
        # Velocidade inicial aleatória (pode variar entre os limites)
        self.velocity = np.random.uniform(bounds[0], bounds[1], n)
        
        # Memória da melhor posição individual (pbest)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')  # Minimização: inicializa com infinito
        
        # Pontuação atual
        self.score = float('inf')