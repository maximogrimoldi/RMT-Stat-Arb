"""
Señales de trading usando Random Matrix Theory (RMT).
Recibe retornos, devuelve señales. No sabe nada de dinero ni fechas.
"""

import os
import numpy as np
import pandas as pd


class RMTStrategy:
    """Pipeline RMT puro: matemática sin lógica de trading."""

    def correlacion(self, R):
        return np.corrcoef(R.T)  # (N, N)

    def pca(self, corr):
        """eigh asume simetría → más estable que eig para matrices de correlación."""
        autovalores, autovectores = np.linalg.eigh(corr)
        idx = np.argsort(autovalores)[::-1]
        return autovalores[idx], autovectores[:, idx]

    def filtrar_marchenko_pastur(self, autovalores, autovectores, T, N):
        """Los autovalores bajo λ_max son ruido de Wishart puro — se descartan."""
        lambda_max = (1 + np.sqrt(N / T)) ** 2
        return autovectores[:, autovalores > lambda_max]  # (N, k)

    def betas(self, R, V):
        """F = R @ V  →  B = (F'F)^{-1} F'R. Devuelve F (T, k) y B (k, N)."""
        F = R @ V
        B = np.linalg.pinv(F.T @ F) @ F.T @ R
        return F, B

    def residuos(self, R, F, B):
        """Lo que no explican los factores — señal de reversión."""
        return R - F @ B  # (T, N)

    def zscore(self, residuos_acum):
        """
        Z-score del último día respecto a toda la historia acumulada.
        Usar la historia completa (no una ventana fija) es la razón para persistir los residuos.
        """
        media  = residuos_acum.mean(axis=0)
        desvio = residuos_acum.std(axis=0)
        desvio[desvio == 0] = np.nan
        return (residuos_acum[-1] - media) / desvio  # (N,)

    def calcular_residuos_rolling(self, retornos, ventana=252):
        """
        Calcula residuos diarios sin lookahead bias.

        Para cada día t, estima betas usando solo retornos[t-ventana:t] (pasado puro),
        luego proyecta el día t con esos betas para obtener el residuo de ese día.

        Por qué rolling y no ventana fija desde el inicio:
        Los factores del mercado cambian con el tiempo — un régimen de alta volatilidad
        genera eigenvectores distintos a uno de baja volatilidad. Si estimáramos betas
        una sola vez con todos los datos, los residuos del año 1 estarían contaminados
        por la estructura del año 3. Rolling mantiene siempre `ventana` días de historia
        relevante y recalibra el modelo a medida que el mercado cambia.

        Devuelve (residuos_df, autovalores_ultimo, lambda_max_ultimo).
        Los primeros `ventana` días quedan como NaN — no hay historia suficiente.
        """
        R       = retornos.values
        tickers = list(retornos.columns)
        T, N    = R.shape

        residuos           = np.full((T, N), np.nan)
        autovalores_ultimo = None
        lambda_max_ultimo  = None

        for t in range(ventana, T):
            R_train = R[t - ventana : t]  # (ventana, N) — solo pasado, nunca el día t

            corr             = self.correlacion(R_train)
            autovalores, autovectores = self.pca(corr)
            V                = self.filtrar_marchenko_pastur(autovalores, autovectores, ventana, N)

            if V.shape[1] == 0:
                continue

            _, B     = self.betas(R_train, V)
            f_t      = R[t] @ V        # factores del día t proyectados con V del pasado
            residuos[t] = R[t] - f_t @ B

            autovalores_ultimo = autovalores
            lambda_max_ultimo  = (1 + np.sqrt(N / ventana)) ** 2

        residuos_df = pd.DataFrame(residuos, index=retornos.index, columns=tickers)
        return residuos_df, autovalores_ultimo, lambda_max_ultimo

    def plot_autovalores(self, autovalores, lambda_max, path):
        """Guarda un gráfico de barras de los autovalores con el umbral Marchenko-Pastur."""
        import matplotlib.pyplot as plt

        colores = ["green" if v > lambda_max else "lightgray" for v in autovalores]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(autovalores)), autovalores, color=colores, width=1.0)
        ax.axhline(
            lambda_max, color="red", linewidth=1.5, linestyle="--",
            label=f"λ_max Marchenko-Pastur = {lambda_max:.2f}",
        )
        ax.set_xlabel("Autovalor (índice, mayor a menor)")
        ax.set_ylabel("Magnitud")
        ax.set_title("Autovalores de la matriz de correlación — verde = señal, gris = ruido")
        ax.legend()

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def test_adf(self, serie, significance=0.05):
        """
        Test Augmented Dickey-Fuller sobre el residuo acumulado de un ticker.
        Devuelve (passed: bool, p_value: float).
        passed=True → residuo estacionario → mean reversion tiene fundamento estadístico.
        passed=False → no hay evidencia de reversión → operar con cautela.
        """
        from statsmodels.tsa.stattools import adfuller
        serie_limpia = serie.dropna()
        if len(serie_limpia) < 20:
            return False, np.nan
        p_value = float(adfuller(serie_limpia)[1])
        return p_value < significance, round(p_value, 4)

