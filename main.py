import numpy as np

def rastrigin(x, n, bounds):
    """
    Evaluates the Rastrigin function for an n-dimensional array x.
    The global minimum is at x = [0, 0, ..., 0] with a value of 0.
    """
    x = np.atleast_1d(x)
    x = np.clip(x, bounds[0], bounds[1])
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def main():
    # Configurações do problema
    n = 2
    bounds = (-5.12, 5.12)
    
    print(f"Preparando PSO para a função Rastrigin com n={n} e limites {bounds}")
    
    # Testando a função Rastrigin (fitness)
    print("\n--- Testes da Função Rastrigin ---")
    test_points = [
        np.array([0.0, 0.0]),         # Mínimo global (resultado esperado: 0.0)
        np.array([-3.12, 3.12]),       # Extremos do limite
        np.array([6.0, -10.0]),       # Fora dos limites (será clipado para 5.12, -5.12)
    ]
    
    for pt in test_points:
        fitness = rastrigin(pt, n, bounds)
        print(f"Ponto: {pt} -> Fitness = {fitness:.4f}")
        
    # TODO: Implementar algoritmo PSO

if __name__ == "__main__":
    main()