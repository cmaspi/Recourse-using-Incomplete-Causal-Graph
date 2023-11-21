import numpy as np
from typing import Callable


class Diamond:
    def __init__(self,
                 n_w: Callable[[], float],
                 n_x: Callable[[], float],
                 n_y: Callable[[], float],
                 n_z: Callable[[], float]
                 ) -> None:
        """_summary_

        Args:
            n_w (Callable[[], float]): _description_
            n_x (Callable[[], float]): _description_
            n_y (Callable[[], float]): _description_
            n_z (Callable[[], float]): _description_
        """
        self.n_w = n_w
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

    def generate_diamond(self,
                         n: int) -> np.ndarray:
        """_summary_

        Args:
            n (int): _description_

        Returns:
            np.ndarray: _description_
        """
        return np.c_[[self.__generate_diamond_helper__() for _ in range(n)]]

    def __generate_diamond_helper__(self
                                    ) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: [w, x, y, z]
        """
        w = self.n_w()
        x = w * w + self.n_x()
        y = 4 * np.sqrt(np.abs(w)) + self.n_y()
        z = 2 * np.sin(x) + 2 * np.sin(y) + self.n_z()
        return np.array([w, x, y, z])

class Lin4V:
    def __init__(self,
                 n_a: Callable[[], float],
                 n_b: Callable[[], float],
                 n_c: Callable[[], float],
                 n_d: Callable[[], float],             
                 ) -> None:
        """_summary_

        Args:
            n_a (Callable[[], float]): _description_
            n_b (Callable[[], float]): _description_
            n_c (Callable[[], float]): _description_
            n_d (Callable[[], float]): _description_
        """
        self.n_a = n_a
        self.n_b = n_b
        self.n_c = n_c
        self.n_d = n_d
        
    def generate_lin4v(self,
                         n: int) -> np.ndarray:
        """_summary_

        Args:
            n (int): _description_

        Returns:
            np.ndarray: _description_
        """
        return np.c_[[self.__generate_lin4v_helper__() for _ in range(n)]]

    def __generate_lin4v_helper__(self
                                    ) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: [w, x, y, z]
        """
        a = self.n_a()
        b = 4*a + self.n_b()
        c = 16*np.sqrt(np.abs(a)) + self.n_c()
        d = 2*np.log(np.abs(c)+1e-6)+self.n_d()
        return np.array([a, b, c, d])

class GermanCredit:
    def __init__(self,
                 n_a: Callable[[], float],
                 n_b: Callable[[], float],
                 n_c: Callable[[], float],
                 n_d: Callable[[], float],
                 n_e: Callable[[], float],
                 n_f: Callable[[], float],
                 n_g: Callable[[], float],             
                 ) -> None:
        """_summary_

        Args:
            n_a (Callable[[], float]): _description_
            n_b (Callable[[], float]): _description_
            n_c (Callable[[], float]): _description_
            n_d (Callable[[], float]): _description_
        """
        self.n_a = n_a
        self.n_b = n_b
        self.n_c = n_c
        self.n_d = n_d
        self.n_e = n_e
        self.n_f = n_f
        self.n_g = n_g
        
    def generate_german_credit(self,
                         n: int) -> np.ndarray:
        """_summary_

        Args:
            n (int): _description_

        Returns:
            np.ndarray: _description_
        """
        return np.c_[[self.__generate_german_credit_helper__() for _ in range(n)]]

    def __generate_german_credit_helper__(self
                                    ) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: [w, x, y, z]
        """
        # Gender
        a = self.n_a()
        # Age
        b = -35 + self.n_b()
        # Education
        c = -0.5 + (1 + np.exp(-(-1 + 0.5*a + 1*(1 + np.exp(- .1*(b)))**(-1) + self.n_c())))**(-1)
        # Loan amount
        d = 1 + .01 * (b - 5) * (5 - b) + 1 * a + self.n_d(),
        d = d[0]
        # Loan duration
        e = -1 + .1 * b + 2 * a + 1 * d + self.n_e(),
        e = e[0]
        # Income
        f = -4 + .1 * (b + 35) + 2 * a + 1 * a * c + self.n_f(),
        f = f[0]
        # Savings
        g = -4 + 1.5 * (f > 0) * f + self.n_g(),
        g = g[0]
        return np.array([a, b, c, d, e, f, g])
    
if __name__ == '__main__':
    DataGen = Diamond(lambda: np.random.uniform(-3, 3),
                      lambda: np.random.uniform(-1, 1),
                      lambda: np.random.uniform(-1, 1),
                      lambda: np.random.uniform(-1, 1)
                      )
    dataset = DataGen.generate_diamond(100)
    print(dataset.shape)
    print(dataset)
