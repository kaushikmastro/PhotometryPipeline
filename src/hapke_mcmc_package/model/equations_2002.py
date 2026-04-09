import numpy as np

# Define constants for clarity and maintainability
MIN_ROUGHNESS_CLIP = 0.1


class HapkeModel:
    """
    Implementation of the Hapke (2002, 2012) photometric model.
    Encapsulates the physics and math for airless body reflectance, including
    the opposition effect and macroscopic roughness.
    """

    def __init__(self, name: str = "Vesta_Regolith"):
        """
        Initializes the Hapke model.

        Args:
            name (str): An optional name for the model instance.
        """
        self.name = name

    def h_function(self, x: float | np.ndarray, w: float) -> float | np.ndarray:
        """
        Ambartsumian-Chandrasekhar H-function for multiple scattering.

        Args:
            x (Union[float, np.ndarray]): The cosine of the incidence or emission angle.
            w (float): Single-scattering albedo (0 <= w <= 1).

        Returns:
            Union[float, np.ndarray]: The value of the H-function.
        """
        if not (0 <= w <= 1):
            raise ValueError("Single-scattering albedo 'w' must be between 0 and 1.")
        gamma = np.sqrt(1 - w)
        return (1 + 2 * x) / (1 + 2 * x * gamma)

    def phase_function(self, alpha: float | np.ndarray, g: float) -> float | np.ndarray:
        """
        Single Particle Phase Function (Henyey-Greenstein).

        Args:
            alpha (Union[float, np.ndarray]): Phase angle in radians.
            g (float): Asymmetry parameter (-1 < g < 1).

        Returns:
            Union[float, np.ndarray]: The value of the phase function.
        """
        if not (-1 < g < 1):
            raise ValueError("Asymmetry parameter 'g' must be between -1 and 1.")
        cos_alpha = np.cos(alpha)
        return (1 - g**2) / (1 + 2 * g * cos_alpha + g**2) ** 1.5

    def shoemaker_opposition_effect(
        self, alpha: float | np.ndarray, B0: float, h: float
    ) -> float | np.ndarray:
        """
        Calculates the Shoemaker opposition effect term.

        Args:
            alpha (Union[float, np.ndarray]): Phase angle in radians.
            B0 (float): Opposition effect amplitude.
            h (float): Opposition effect width parameter.

        Returns:
            Union[float, np.ndarray]: The opposition effect multiplier.
        """
        # Ensure alpha is non-negative for the tan function
        g = np.abs(alpha)
        return 1 + B0 / (1 + np.tan(g / 2) / h)

    def macroscopic_roughness(
        self, i: float | np.ndarray, e: float | np.ndarray, theta_bar: float
    ) -> float | np.ndarray:
        """
        The S(i, e, theta) term for macroscopic roughness.

        Args:
            i (Union[float, np.ndarray]): Incidence angle in radians.
            e (Union[float, np.ndarray]): Emission angle in radians.
            theta_bar (float): Mean slope angle for roughness in radians.

        Returns:
            Union[float, np.ndarray]: The macroscopic roughness factor.
        """
        tan_theta = np.tan(theta_bar)
        # The formula can lead to division by zero or very large numbers if i or e are near pi/2.
        # We avoid this by taking the max of i and e.
        S = 1.0 / (1.0 + tan_theta * np.tan(np.maximum(i, e)))
        return np.clip(S, MIN_ROUGHNESS_CLIP, 1.0)

    def compute_reflectance(
        self,
        i: float | np.ndarray,
        e: float | np.ndarray,
        alpha: float | np.ndarray,
        w: float,
        g: float,
        theta_bar: float,
        B0: float,
        h: float,
    ) -> float | np.ndarray:
        """
        Calculates I/F (Reflectance) using the full Hapke model.

        Args:
            i (Union[float, np.ndarray]): Incidence angle in radians.
            e (Union[float, np.ndarray]): Emission angle in radians.
            alpha (Union[float, np.ndarray]): Phase angle in radians.
            w (float): Single-scattering albedo.
            g (float): Asymmetry parameter.
            theta_bar (float): Macroscopic roughness angle in radians.
            B0 (float): Opposition effect amplitude.
            h (float): Opposition effect width.

        Returns:
            Union[float, np.ndarray]: The calculated reflectance (I/F).
        """
        mu0 = np.cos(i)
        mu = np.cos(e)

        # Ensure angles are within valid physical range to avoid math errors
        if np.any(mu0 <= 0) or np.any(mu <= 0):
            # Reflectance is zero if light is at or below the horizon
            return np.zeros_like(i)

        pf = self.phase_function(alpha, g)
        B_sh = self.shoemaker_opposition_effect(alpha, B0, h)

        h_i = self.h_function(mu0, w)
        h_e = self.h_function(mu, w)
        multi_scattering = (h_i * h_e) - 1

        S = self.macroscopic_roughness(i, e, theta_bar)

        # Full Hapke equation
        reflectance = (w / 4) * (mu0 / (mu0 + mu)) * ((pf * B_sh) + multi_scattering) * S

        return reflectance
