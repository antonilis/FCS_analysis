import numpy as np


# ==== FUNKCJE MODELI ====

def G_1comp(tau, N, tau_D, kappa):
    return (1 / N) * (1 / (1 + tau / tau_D)) * (1 / np.sqrt(1 + tau / (kappa ** 2 * tau_D)))


def G_1comp_triplet(tau, N, T, tau_trip, tau_D, kappa):
    triplet_term = (1 + T * np.exp(-tau / tau_trip)) / (1 - T)
    return (1 / N) * triplet_term * (1 / (1 + tau / tau_D)) * (1 / np.sqrt(1 + tau / (kappa ** 2 * tau_D)))


def G_2comp(tau, N, A, tau_D1, tau_D2, kappa):
    term1 = A / ((1 + tau / tau_D1) * np.sqrt(1 + tau / (kappa ** 2 * tau_D1)))
    term2 = (1 - A) / ((1 + tau / tau_D2) * np.sqrt(1 + tau / (kappa ** 2 * tau_D2)))
    return (1 / N) * (term1 + term2)


def G_2comp_triplet(tau, N, A, tau_D1, tau_D2, kappa, T, tau_trip):
    triplet_term = (1 + T * np.exp(-tau / tau_trip)) / (1 - T)
    term1 = A / ((1 + tau / tau_D1) * np.sqrt(1 + tau / (kappa ** 2 * tau_D1)))
    term2 = (1 - A) / ((1 + tau / tau_D2) * np.sqrt(1 + tau / (kappa ** 2 * tau_D2)))
    return (1 / N) * triplet_term * (term1 + term2)


def G_anomalous(tau, N, tau_D, kappa, alpha):
    return (1 / N) * (1 / ((1 + (tau / tau_D) ** alpha) * np.sqrt(1 + (tau / (kappa ** 2 * tau_D)) ** alpha)))


def G_3comp(tau, N, A1, A2, tau_D1, tau_D2, tau_D3, kappa):
    G1 = A1 / ((1 + tau / tau_D1) * np.sqrt(1 + tau / (kappa ** 2 * tau_D1)))
    G2 = A2 / ((1 + tau / tau_D2) * np.sqrt(1 + tau / (kappa ** 2 * tau_D2)))
    G3 = (1 - A1 - A2) / ((1 + tau / tau_D3) * np.sqrt(1 + tau / (kappa ** 2 * tau_D3)))
    return (1 / N) * (G1 + G2 + G3)


def G_3comp_triplet(tau, N, A1, A2, tau_D1, tau_D2, tau_D3, kappa, T, tau_trip):
    triplet_term = (1 + T * np.exp(-tau / tau_trip)) / (1 - T)
    G1 = A1 / ((1 + tau / tau_D1) * np.sqrt(1 + tau / (kappa ** 2 * tau_D1)))
    G2 = A2 / ((1 + tau / tau_D2) * np.sqrt(1 + tau / (kappa ** 2 * tau_D2)))
    G3 = (1 - A1 - A2) / ((1 + tau / tau_D3) * np.sqrt(1 + tau / (kappa ** 2 * tau_D3)))
    return (1 / N) * triplet_term * (G1 + G2 + G3)


def G_2comp_trans_rot(tau, G_inf, Np, q, A2f, A4f, eta_rot, T, eta_T1, eta_T2, R1, R2, r1, r2, omega_0, kappa):
    # Stała Boltzmanna
    kB = 1.380649e-23  # J/K

    # Składniki obrotowe GR1
    a2 = (4 * A2f * eta_rot * R1) / (r1 ** 3)
    a4 = (5 * A4f * eta_rot * R1) / (r1 ** 3)
    GR1 = 1 + A2f * np.exp(-a2 * tau) + A4f * np.exp(-a4 * tau)

    # Czas dyfuzji translacyjnej T1 i T2
    tau_T1 = (3 * T * eta_T1 * omega_0 ** 2) / (kB * r1)
    tau_T2 = (3 * T * eta_T2 * omega_0 ** 2) / (kB * r2)

    # Składniki translacyjne
    GT1 = 1 / ((1 + tau / tau_T1) * np.sqrt(1 + tau / (kappa ** 2 * tau_T1)))
    GT2 = 1 / ((1 + tau / tau_T2) * np.sqrt(1 + tau / (kappa ** 2 * tau_T2)))

    # Pełna funkcja korelacji
    return G_inf + (1 / Np) * (q * GR1 * GT1 + (1 - q) * GT2)


# ==== MAPOWANIE NAZWY -> FUNKCJA ====

FCS_fitting_functions = {
    "One-component simple diffusion": G_1comp,
    'One-component diffusion with triplets states': G_1comp_triplet,
    "Two-component simple diffusion": G_2comp,
    "Two-component diffusion with triplet states": G_2comp_triplet,
    "Anomalous diffusion": G_anomalous,
    "Three-component simple diffusion": G_3comp,
    "Three-component simple diffusion with triplets": G_3comp_triplet,
    "Two-component diffusion: 1st translational diffusion only, 2nd Translational and rotational diffusion": G_2comp_trans_rot
}
