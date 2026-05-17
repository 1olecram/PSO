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

    def update_velocity(self, global_best_position, c1, c2, w):
        cognitive_component = c1 * np.random.uniform(0, 1) * (self.best_position - self.position)
        social_component = c2 * np.random.uniform(0, 1) * (global_best_position - self.position)
        self.velocity = self.constriction_factor(c1, c2) * (w * self.velocity + cognitive_component + social_component)
    
    def update_position(self, bounds):
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1]) # Limita as posições dentro dos limites do problema (tudo abaxio de 
                                                                     # -5.12 vira -5.12 e tudo acima de 5.12 vira 5.12)
    def constriction_factor(c1, c2):
        phi = c1 + c2
        return 2 / (2 - phi - np.sqrt(phi**2 - 4 * phi))