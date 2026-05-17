from Particle import Particle
import numpy as np

def rastrigin(x, n, bounds):
    """
    Avalia a função Rastrigin para um array x de n dimensões.
    O mínimo global está em x = [0, 0, ..., 0] com um valor de 0.
    """
    x = np.atleast_1d(x)
    x = np.clip(x, bounds[0], bounds[1]) # limitar posição dentro dos limites do problema (tudo abaixo de -5.12 vira -5.12 e tudo acima de 5.12 vira 5.12)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)

def weighting_inertia(i, w_max =0.9, w_min = 0.4, iter_max = 100):
    """
    Calcula o valor do parâmetro de inércia (w) para a iteração atual.
    Decaimento linear de w_max para w_min ao longo das iterações.
    """
    return w_max - i/iter_max * (w_max - w_min)

def main():

    n = 2 # Dimensões do problema
    bounds = (-5.12, 5.12) # Limites do problema
    c1 = 2
    c2 = 2
        
    # Inicializando o Enxame (Swarm)
    num_particles = 30
    swarm = [Particle(n, bounds) for _ in range(num_particles)]

    # Inicializa o gbest
    global_best_position = np.zeros(n)
    global_best_score = float('inf')

    iter_max = 100
    
    # Laço de evolução do PSO
    for i in range(iter_max):
        w = weighting_inertia(i, iter_max=iter_max)
        
        # Avaliação e atualização de pbest e gbest
        for particle in swarm:
            # Avalia a função Rastrigin
            score = rastrigin(particle.position, n, bounds)
            particle.score = score
            
            # Atualiza o best position (pbest)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = np.copy(particle.position)
                
            # Atualiza o global best position (gbest)
            if score < global_best_score:
                global_best_score = score
                global_best_position = np.copy(particle.position)
                
        # Movimentação (atualização de velocidade e posição)
        for particle in swarm:
            particle.update_velocity(global_best_position, c1, c2, w)
            particle.update_position(bounds)

    print(f"\n--- Resultado Final após {iter_max} iterações ---")
    print(f"Melhor pontuação (gbest_score): {global_best_score}")
    print(f"Melhor posição (gbest_position): {global_best_position}")
if __name__ == "__main__":
    main()