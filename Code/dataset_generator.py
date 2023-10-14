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
        x = w*w + self.n_x()
        y = 4*np.sqrt(np.abs(w)) + self.n_y()
        z = 2*np.sin(x) + 2*np.sin(y) + self.n_z()
        return np.array([w, x, y, z])


if __name__ == '__main__':
    DataGen = Diamond(lambda: np.random.uniform(-3, 3),
                      lambda: np.random.uniform(-1, 1),
                      lambda: np.random.uniform(-1, 1),
                      lambda: np.random.uniform(-1, 1)
                      )
    dataset = DataGen.generate_diamond(100)
    print(dataset.shape)
    print(dataset)
    