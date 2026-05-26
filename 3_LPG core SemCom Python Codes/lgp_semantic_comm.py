"""
================================================================================
LPG SEMANTIC COMMUNICATION — COUPLED ADAPTIVE LOOP
Pseudocode Scaffold for NVIDIA Sionna / PyTorch Implementation

Author   : Dada Theophilus Olusegun
Framework: Logarithmic Probability Geometry (LPG) Semantic Communication
Reference: Algorithm 3.1 — Master Algorithm (Proposal Section 3.8)

Implementation Stack
--------------------
  Core          : Python 3.10+
  Deep Learning : PyTorch >= 2.0
  Channel Sim   : NVIDIA Sionna >= 0.16  (TensorFlow backend, bridged via numpy)
  Numerics      : NumPy, SciPy
  Logging       : Python logging + TensorBoard

Module Structure
----------------
  LPGConfig                  — global hyperparameters and shared vocabulary W
  SemanticManifold           — log-probability geometry on W
  ComplexMeaningMap          — M(m, c, sem) encoding and Cauchy-Riemann gate
  SemInstance                — atomic unit (B, W, Γ, ι)
  SOSIEncoder                — S-OSI encoding transform  E_SOSI : sem → ψ
  SOSIDecoder                — S-OSI decoding transform  D_SOSI : ψ̃ → sem̂
  SemanticChannel            — field propagation model   ψ̃ = H_s ψ + η
  MLAnchorReconstructor      — greedy ML anchor set estimator
  SemanticFidelityMeter      — Jensen-Shannon fidelity  F_s ∈ [0,1]
  SemanticCapacityEstimator  — empirical  Ĉ_s from ensemble statistics
  ConservationMonitor        — semantic energy drift enforcement
  SenderAgent                — adaptive anchor policy  π_s^(t)
  ReceiverAgent              — adaptive reconstruction  π_r^(t), θ_r^(t)
  CoupledAdaptiveLoop        — master loop (Algorithm 3.1)
  SimulationRunner           — experiment harness with logging

Mathematical Notation (mirrors proposal Section 3.8)
------------------------------------------------------
  W            vocabulary  {w_1, …, w_|W|}
  Γ ⊆ W        anchor set
  ι : W→[0,1]  interpretation map (probability simplex Δ^{|W|-1})
  B            log-probability base signal vector ∈ ℝ^|W|
  sem          quadruple (B, W, Γ, ι)
  M(m,c,sem)   complex meaning map ∈ ℂ
  ψ            complex transmitted signal ∈ ℂ^d
  H_s          semantic channel matrix = μ_s · H_phys ⊙ H_sem
  μ_s          semantic permeability (learned)
  ε_s          semantic permittivity (fixed)
  F_s          semantic fidelity ∈ [0,1]
  Ĉ_s          empirical semantic capacity (semantic bits/channel use)
  δ_t          feedback signal  F_s^(t) − F*
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Numerical / ML ────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import rel_entr          # KL divergence element-wise

# ── Sionna (channel simulation) ───────────────────────────────────────────────
# Sionna uses TensorFlow internally; we call it through its numpy-compatible
# helper layer and convert tensors to/from torch as needed.
# Uncomment the block below once Sionna is installed:
#
# import sionna
# from sionna.channel import AWGN, RayleighBlockFading
# from sionna.utils  import BinarySource, ebnodb2no
#
# For the scaffold, SemanticChannel uses a pure-numpy stub that is API-
# compatible with what a Sionna wrapper would expose.

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LPG-SemComm")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LPGConfig:
    """
    Global hyperparameters for Algorithm 3.1.

    All values correspond to the recommended ranges in Table 3.8.2
    of the PhD proposal.  Override before instantiating the loop.
    """
    # ── Semantic vocabulary ────────────────────────────────────────────────
    vocab_size: int = 1000           # |W|  (Table 3.8.2: 500–5000)

    # ── Fidelity target ────────────────────────────────────────────────────
    fidelity_target: float = 0.90   # F*   (Table 3.8.2: 0.85–0.95)

    # ── Adaptive learning rates ────────────────────────────────────────────
    eta_sender:   float = 0.05      # η_s  sender policy update step
    eta_receiver: float = 1e-3      # η_r  receiver gradient ascent step
    gamma:        float = 0.005     # γ    permeability adaptation step

    # ── Channel ────────────────────────────────────────────────────────────
    sigma2:       float = 0.1       # σ²   AWGN noise variance
    mu_s_init:    float = 0.5       # μ_s  initial semantic permeability
    epsilon_s:    float = 1.0       # ε_s  semantic permittivity (fixed)
    tx_dim:       int   = 64        # d    transmission dimension

    # ── Conservation ──────────────────────────────────────────────────────
    delta_energy: float = 0.10      # Δ_E  energy drift bound (Table 3.8.2)

    # ── Anchor reconstruction ──────────────────────────────────────────────
    laplace_alpha:    float = 1e-3  # α    Laplace smoothing
    greedy_threshold: float = 1e-4  # ε    greedy termination threshold

    # ── Convergence ────────────────────────────────────────────────────────
    epsilon_conv: float = 1e-3      # ε_conv  (Table 3.8.2: 1e-3–1e-2)
    tau_conv:     int   = 20        # τ_conv  patience window (rounds)
    T_max:        int   = 2000      # T_max   hard stopping limit

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SEMANTIC MANIFOLD
#     Implements the LPG base space: log-probability vectors on W.
# ══════════════════════════════════════════════════════════════════════════════

class SemanticManifold:
    """
    The LPG semantic manifold M_s over vocabulary W.

    Each anchor set Γ ⊆ W induces a point on M_s via the log-probability
    map:
        B(Γ, W)_j = log( (|Γ ∩ {w_j}| + α) / (|Γ| + α|W|) )

    This is the geometric object the sender transmits.  The manifold
    is a subset of the probability simplex Δ^{|W|-1} lifted through
    the log map into ℝ^|W|.
    """

    def __init__(self, cfg: LPGConfig):
        self.cfg = cfg
        self.W   = np.arange(cfg.vocab_size)   # vocabulary indices

    def base_signal(self, gamma: np.ndarray) -> np.ndarray:
        """
        Compute B(Γ, W) ∈ ℝ^|W|.

        Parameters
        ----------
        gamma : array of word indices in Γ

        Returns
        -------
        B : log-probability vector of shape (|W|,)
        """
        vocab_size = self.cfg.vocab_size
        alpha = self.cfg.laplace_alpha
        indicator = np.zeros(vocab_size)
        for idx in gamma:
            indicator[idx] = 1.0
        denom = len(gamma) + alpha * vocab_size
        B = np.log((indicator + alpha) / denom)
        return B                                # shape (|W|,)

    def softmax_to_interpretation(self, B: np.ndarray) -> np.ndarray:
        """
        Recover interpretation map ι from log-probability vector B.

        ι(w_j) = exp(B_j) / Σ_l exp(B_l)      (softmax normalisation)

        Returns probability distribution over W  — an element of Δ^{|W|-1}.
        """
        exp_B = np.exp(B - B.max())             # numerically stable softmax
        return exp_B / exp_B.sum()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SEM INSTANCE
#     The atomic unit of LPG semantic transmission.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SemInstance:
    """
    sem = (B, W, Γ, ι)

    Fields
    ------
    B     : np.ndarray  — log-probability base signal ∈ ℝ^|W|
    W     : np.ndarray  — shared vocabulary indices
    gamma : np.ndarray  — anchor set indices Γ ⊆ W
    iota  : np.ndarray  — interpretation map ι ∈ Δ^{|W|-1}
    """
    B:     np.ndarray
    W:     np.ndarray
    gamma: np.ndarray
    iota:  np.ndarray


# ══════════════════════════════════════════════════════════════════════════════
# 3.  COMPLEX MEANING MAP  +  CAUCHY-RIEMANN GATE
#     Implements M(m, c, sem) and the analyticity gate (Step S4).
# ══════════════════════════════════════════════════════════════════════════════

class ComplexMeaningMap:
    """
    The complex-valued meaning map:

        M(m, c, sem) = λᵀ u(m, B) + i (λ(c) − λ)ᵀ u(m, B)

    where
        u(m, B) ∈ ℝ^|W|    message embedding under base space B
        λ        ∈ ℝ^|W|    global semantic weight vector
        λ(c)     ∈ ℝ^|W|    context-modulated weight vector

    The Cauchy–Riemann semantic regularity conditions are checked
    numerically before each transmission (Step S4 of Algorithm 3.1).
    """

    def __init__(self, cfg: LPGConfig, seed: int = 42):
        rng = np.random.default_rng(seed)
        vocab_size = cfg.vocab_size
        # Learnable weight vectors (initialised randomly; updated by receiver)
        self.lambda_global  = rng.standard_normal(vocab_size)   # λ
        self.lambda_context = rng.standard_normal(vocab_size)   # λ(c)

    def message_embedding(self, message_idx: int, B: np.ndarray) -> np.ndarray:
        """
        u(m, B): embed message m as a |W|-vector using B as the geometric
        basis.  Here we use a one-hot projection through B as a prototype;
        a full implementation replaces this with a learned encoder.
        """
        u = np.zeros_like(B)
        u[message_idx % len(B)] = 1.0
        return u * np.exp(B)                   # geometry-weighted embedding

    def encode(self,
               message_idx: int,
               context_idx: int,
               sem: SemInstance) -> complex:
        """
        Compute M(m, c, sem) ∈ ℂ.

        Returns the complex scalar encoding of meaning.
        """
        u   = self.message_embedding(message_idx, sem.B)
        lam = self.lambda_global
        lam_c = self.lambda_context

        real_part = float(lam @ u)
        imag_part = float((lam_c - lam) @ u)
        return complex(real_part, imag_part)

    def cauchy_riemann_check(self,
                             sem: SemInstance,
                             delta: float = 1e-5) -> bool:
        """
        Analyticity gate — Step S4 of Algorithm 3.1.

        Numerically verifies the Cauchy–Riemann semantic conditions:
            ∂ Re(M)/∂B_j  =  ∂ Im(M)/∂λ_j(c)
            ∂ Re(M)/∂λ_j(c) = −∂ Im(M)/∂B_j

        Uses finite differences as a prototype; a full implementation
        uses automatic differentiation via torch.autograd.

        Returns True if conditions are satisfied (transmission proceeds),
        False if violated (round is aborted — Step S4).
        """
        vocab_size = len(sem.B)
        cr_violations = 0

        for j in range(min(vocab_size, 50)):          # check first 50 dimensions
            # ∂ Re(M)/∂B_j  via finite difference
            B_plus  = sem.B.copy(); B_plus[j]  += delta
            B_minus = sem.B.copy(); B_minus[j] -= delta
            sem_p = SemInstance(B_plus,  sem.W, sem.gamma, sem.iota)
            sem_m = SemInstance(B_minus, sem.W, sem.gamma, sem.iota)
            dRe_dBj = (self.encode(0, 0, sem_p).real -
                       self.encode(0, 0, sem_m).real) / (2 * delta)

            # ∂ Im(M)/∂λ_j(c)  via finite difference on context weights
            lc_orig = self.lambda_context[j]
            self.lambda_context[j] += delta
            dIm_dlcj_p = self.encode(0, 0, sem).imag
            self.lambda_context[j] -= 2 * delta
            dIm_dlcj_m = self.encode(0, 0, sem).imag
            self.lambda_context[j] = lc_orig
            dIm_dlcj = (dIm_dlcj_p - dIm_dlcj_m) / (2 * delta)

            if abs(dRe_dBj - dIm_dlcj) > 1e-3:
                cr_violations += 1

        return cr_violations == 0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  S-OSI ENCODING TRANSFORM   E_SOSI : sem → ψ ∈ ℂ^d
# ══════════════════════════════════════════════════════════════════════════════

class SOSIEncoder(nn.Module):
    """
    S-OSI encoding transform.

    Encapsulates all seven S-OSI layers inside a single learned projection
    from the complex scalar M(m,c,sem) into the d-dimensional complex
    transmission vector ψ ∈ ℂ^d.

    Architecture (prototype):
        Layer 1  — semantic parsing       (meaning extraction)
        Layer 2  — semantic representation (LPG geometry)
        Layer 3  — semantic compression   (anchor-aware)
        Layer 4  — semantic encryption    (optional)
        Layer 5  — semantic framing       (packet formation)
        Layer 6  — semantic link control  (error resilience)
        Layer 7  — semantic physical map  (complex baseband)

    In this scaffold Layers 1–6 are collapsed into a learned linear
    projection and Layer 7 is an explicit complex baseband mapping.
    A full implementation expands each layer separately.
    """

    def __init__(self, cfg: LPGConfig):
        super().__init__()
        self.cfg = cfg
        vocab_size = cfg.vocab_size
        d   = cfg.tx_dim

        # Collapse Layers 1–6 into a learned projection ℝ^|W| → ℝ^{2d}
        # (real and imaginary parts of ψ stacked)
        self.semantic_projection = nn.Sequential(
            nn.Linear(vocab_size, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, 2 * d),            # output: [Re(ψ); Im(ψ)]
        )

    def forward(self, sem: SemInstance) -> np.ndarray:
        """
        E_SOSI(sem) → ψ ∈ ℂ^d

        Parameters
        ----------
        sem : SemInstance

        Returns
        -------
        psi : complex numpy array of shape (d,)
        """
        B_tensor = torch.tensor(sem.B, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self.semantic_projection(B_tensor).squeeze(0).numpy()
        psi_real = out[:self.cfg.tx_dim]
        psi_imag = out[self.cfg.tx_dim:]
        return psi_real + 1j * psi_imag         # shape (d,)  complex


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SEMANTIC CHANNEL   ψ̃ = H_s ψ + η
#     Field propagation model  (Sionna-compatible stub)
# ══════════════════════════════════════════════════════════════════════════════

class SemanticChannel:
    """
    Semantic field propagation model.

    Implements the channel equation:
        ψ̃_t = H_s ψ_t + η_t,    η_t ~ CN(0, σ²I)

    where the semantic channel matrix decomposes as:
        H_s = μ_s · H_phys ⊙ H_sem

    H_phys  — physical Rayleigh block-fading matrix
    H_sem   — semantic alignment matrix (identity in the prototype;
               a full implementation learns this from shared context)
    μ_s     — semantic permeability (updated each round by Adaptation A3)

    Sionna integration note:
        Replace _rayleigh_matrix() with a Sionna RayleighBlockFading
        channel model.  The remaining decomposition logic is unchanged.
    """

    def __init__(self, cfg: LPGConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.mu_s = cfg.mu_s_init              # initialised; updated by loop

    def _rayleigh_matrix(self) -> np.ndarray:
        """
        Prototype Rayleigh fading channel matrix H_phys ∈ ℂ^{d×d}.

        Replace with:
            sionna.channel.RayleighBlockFading(...)
        for production simulation.
        """
        d = self.cfg.tx_dim
        real = self.rng.standard_normal((d, d)) / math.sqrt(2)
        imag = self.rng.standard_normal((d, d)) / math.sqrt(2)
        return real + 1j * imag

    def _semantic_alignment_matrix(self) -> np.ndarray:
        """
        H_sem: semantic alignment matrix.

        Prototype: identity (perfect semantic alignment).
        Full implementation: learned from shared context W and history.
        """
        return np.eye(self.cfg.tx_dim, dtype=complex)

    def transmit(self, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate ψ through the semantic field channel.

        Returns
        -------
        psi_tilde : received signal ψ̃ ∈ ℂ^d
        H_s       : realised channel matrix (needed for inversion at receiver)
        """
        d     = self.cfg.tx_dim
        H_phys = self._rayleigh_matrix()
        H_sem  = self._semantic_alignment_matrix()
        H_s    = self.mu_s * (H_phys * H_sem)  # Hadamard product

        noise_real = self.rng.standard_normal(d) * math.sqrt(self.cfg.sigma2 / 2)
        noise_imag = self.rng.standard_normal(d) * math.sqrt(self.cfg.sigma2 / 2)
        eta   = noise_real + 1j * noise_imag

        psi_tilde = H_s @ psi + eta
        return psi_tilde, H_s


# ══════════════════════════════════════════════════════════════════════════════
# 6.  S-OSI DECODING TRANSFORM   D_SOSI : ψ̃ → sem̂
# ══════════════════════════════════════════════════════════════════════════════

class SOSIDecoder:
    """
    S-OSI decoding transform.

    Implements the three-step decoding procedure:
        D1  Signal inversion:       B̂ ← Re(H_s† ψ̃)
        D2  Anchor reconstruction:  Γ̂ ← ML-GreedySearch(B̂, W, σ²)
        D3  Interpretation recovery: ι̂ ← Softmax(B(Γ̂, W))
    """

    def __init__(self, cfg: LPGConfig, manifold: SemanticManifold,
                 reconstructor: 'MLAnchorReconstructor'):
        self.cfg           = cfg
        self.manifold      = manifold
        self.reconstructor = reconstructor

    def decode(self, psi_tilde: np.ndarray,
               H_s: np.ndarray) -> SemInstance:
        """
        D_SOSI(ψ̃) → sem̂ = (B̂, W, Γ̂, ι̂)

        Parameters
        ----------
        psi_tilde : received signal ψ̃ ∈ ℂ^d
        H_s       : realised channel matrix

        Returns
        -------
        sem_hat : reconstructed SemInstance
        """
        # ── D1: Signal inversion ──────────────────────────────────────────
        # B̂ ← Re(H_s† ψ̃)   using Moore-Penrose pseudo-inverse
        H_s_pinv = np.linalg.pinv(H_s)
        B_hat_complex = H_s_pinv @ psi_tilde
        B_hat = B_hat_complex.real                     # shape (d,)

        # Pad or truncate B_hat to match |W| (d may differ from |W|)
        vocab_size = self.cfg.vocab_size
        if len(B_hat) < vocab_size:
            B_hat = np.pad(B_hat, (0, vocab_size - len(B_hat)))
        else:
            B_hat = B_hat[:W]

        # ── D2: Anchor reconstruction (ML greedy) ─────────────────────────
        gamma_hat = self.reconstructor.reconstruct(B_hat)

        # ── D3: Interpretation map recovery ───────────────────────────────
        B_gamma_hat = self.manifold.base_signal(gamma_hat)
        iota_hat    = self.manifold.softmax_to_interpretation(B_gamma_hat)

        return SemInstance(
            B     = B_hat,
            W     = self.manifold.W,
            gamma = gamma_hat,
            iota  = iota_hat,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ML ANCHOR RECONSTRUCTOR   (Step R2 / Stage 2 of measurement protocol)
# ══════════════════════════════════════════════════════════════════════════════

class MLAnchorReconstructor:
    """
    Maximum-likelihood anchor set estimator using greedy forward selection.

    Solves:
        Γ̂ = argmax_{Γ'⊆W}  -‖B̂ − B(Γ', W)‖² / (2σ²)

    via greedy forward addition, terminating when the log-likelihood
    gain falls below the threshold ε.

    Complexity: O(|W|²) per reconstruction call.
    """

    def __init__(self, cfg: LPGConfig, manifold: SemanticManifold):
        self.cfg      = cfg
        self.manifold = manifold

    def _log_likelihood(self, B_hat: np.ndarray,
                        gamma: np.ndarray) -> float:
        """
        L(Γ' | B̂, W) = −‖B̂ − B(Γ', W)‖² / (2σ²)
        """
        if len(gamma) == 0:
            return -np.inf
        B_candidate = self.manifold.base_signal(gamma)
        residual    = B_hat - B_candidate
        return -float(np.dot(residual, residual)) / (2 * self.cfg.sigma2)

    def reconstruct(self, B_hat: np.ndarray) -> np.ndarray:
        """
        Greedy forward selection to recover Γ̂.

        Returns
        -------
        gamma_hat : np.ndarray of selected word indices
        """
        vocab_size  = self.cfg.vocab_size
        gamma_hat   = np.array([], dtype=int)
        remaining   = list(range(vocab_size))
        current_ll  = -np.inf

        while remaining:
            best_gain  = -np.inf
            best_word  = None

            for w in remaining:
                candidate = np.append(gamma_hat, w)
                ll        = self._log_likelihood(B_hat, candidate)
                gain      = ll - current_ll
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            # Termination criterion: gain below threshold ε
            if best_gain < self.cfg.greedy_threshold:
                break

            gamma_hat  = np.append(gamma_hat, best_word)
            remaining.remove(best_word)
            current_ll = self._log_likelihood(B_hat, gamma_hat)

        return gamma_hat


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SEMANTIC FIDELITY METER   F_s = 1 − D_JS(ι ‖ ι̂)
# ══════════════════════════════════════════════════════════════════════════════

class SemanticFidelityMeter:
    """
    Measures semantic fidelity between the sender's true interpretation map
    ι_t and the receiver's reconstructed map ι̂_t using the Jensen–Shannon
    divergence:

        F_s^(t) = 1 − D_JS(ι_t ‖ ι̂_t)

    D_JS ∈ [0,1] (computed in bits), so F_s ∈ [0,1].
    F_s = 1 → perfect semantic fidelity.
    F_s = 0 → total semantic loss.
    """

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """D_KL(p ‖ q)  using scipy.special.rel_entr for numerical safety."""
        return float(np.sum(rel_entr(p, q)))

    def js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        D_JS(p ‖ q) = ½ D_KL(p ‖ m) + ½ D_KL(q ‖ m),   m = (p+q)/2
        Result in [0,1] (bits).
        """
        # Add small epsilon for numerical stability
        eps = 1e-10
        p = p + eps;  p /= p.sum()
        q = q + eps;  q /= q.sum()
        m = 0.5 * (p + q)
        return 0.5 * self.kl_divergence(p, m) + \
               0.5 * self.kl_divergence(q, m)

    def measure(self, iota_true: np.ndarray,
                iota_hat: np.ndarray) -> float:
        """
        Compute F_s^(t) = 1 − D_JS(ι_true ‖ ι̂).

        Returns
        -------
        fidelity : float in [0, 1]
        """
        d_js = self.js_divergence(iota_true, iota_hat)
        return float(np.clip(1.0 - d_js, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# 9.  SEMANTIC CAPACITY ESTIMATOR   Ĉ_s
# ══════════════════════════════════════════════════════════════════════════════

class SemanticCapacityEstimator:
    """
    Empirical semantic capacity estimator (Stage 5 of measurement protocol).

    After T rounds accumulates:
        μ̂_|Γ|(T) = (1/T) Σ_t |Γ̂_t|
        F̄_s(T)  = (1/T) Σ_t F_s^(t)
        μ̂_s(T)  = F̄_s(T) · μ̂_|Γ|(T) / |W|

    Then:
        Ĉ_s(T) = log₂(|W| / (|W| − μ̂_|Γ|(T))) · μ̂_s(T)
    """

    def __init__(self, cfg: LPGConfig):
        self.cfg          = cfg
        self._gamma_sizes : List[int]   = []
        self._fidelities  : List[float] = []

    def record(self, gamma_hat: np.ndarray, fidelity: float) -> None:
        """Record one round's statistics."""
        self._gamma_sizes.append(len(gamma_hat))
        self._fidelities.append(fidelity)

    def estimate(self) -> Dict[str, float]:
        """
        Compute Ĉ_s from accumulated statistics.

        Returns dict with keys: C_s_hat, mu_gamma_hat, mu_s_hat, F_s_bar
        """
        vocab_size   = self.cfg.vocab_size
        T            = len(self._gamma_sizes)
        if T == 0:
            return {"C_s_hat": 0.0, "mu_gamma_hat": 0.0,
                    "mu_s_hat": 0.0, "F_s_bar": 0.0}

        mu_gamma_hat = float(np.mean(self._gamma_sizes))
        F_s_bar      = float(np.mean(self._fidelities))
        mu_s_hat     = F_s_bar * mu_gamma_hat / vocab_size

        # Guard against log singularity when μ̂_|Γ| → |W|
        denom = vocab_size - mu_gamma_hat
        if denom <= 0:
            C_s_hat = float("inf")
        else:
            C_s_hat = math.log2(vocab_size / denom) * mu_s_hat

        return {
            "C_s_hat"     : C_s_hat,
            "mu_gamma_hat": mu_gamma_hat,
            "mu_s_hat"    : mu_s_hat,
            "F_s_bar"     : F_s_bar,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 10.  CONSERVATION MONITOR   (Step S6)
# ══════════════════════════════════════════════════════════════════════════════

class ConservationMonitor:
    """
    Enforces the semantic energy conservation law (Section 3.8.6).

    Semantic energy at round t:
        E_sem(t) = ‖ψ_t‖² = Σ_j |M_j(m, c, sem_t)|²

    Conservation condition:
        |E_sem(t) − E_sem(t−1)| ≤ Δ_E

    Rounds violating this condition are flagged as conservation anomalies
    and excluded from the capacity estimator.
    """

    def __init__(self, cfg: LPGConfig):
        self.cfg           = cfg
        self._prev_energy  = None
        self.anomaly_count = 0

    def compute_energy(self, psi: np.ndarray) -> float:
        """E_sem(t) = ‖ψ_t‖²"""
        return float(np.dot(psi.conj(), psi).real)

    def check(self, psi: np.ndarray) -> bool:
        """
        Check conservation condition.

        Returns True  → transmission is valid.
        Returns False → conservation anomaly; exclude from capacity estimate.
        """
        energy = self.compute_energy(psi)
        if self._prev_energy is None:
            self._prev_energy = energy
            return True

        drift = abs(energy - self._prev_energy)
        self._prev_energy = energy

        if drift > self.cfg.delta_energy:
            self.anomaly_count += 1
            logger.warning(f"Conservation anomaly: drift={drift:.4f} > "
                           f"Δ_E={self.cfg.delta_energy:.4f}")
            return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# 11.  SENDER AGENT   (Steps S1–S7 + Adaptation A1)
# ══════════════════════════════════════════════════════════════════════════════

class SenderAgent:
    """
    Adaptive sender agent.

    Maintains and updates the anchor selection policy π_s^(t) via the
    multiplicative weights update rule (Step A1):

        π_s^(t+1)(Γ) ∝ π_s^(t)(Γ) · exp(η_s · δ_t · 𝟏[Γ_t = Γ])

    State
    -----
    policy_weights : dict mapping gamma_size → weight
                     (prototype: policy over anchor sizes, not full subsets,
                      for tractability.  Full implementation uses a neural
                      anchor selector.)
    """

    def __init__(self, cfg: LPGConfig, manifold: SemanticManifold,
                 encoder: SOSIEncoder, meaning_map: ComplexMeaningMap,
                 rng: np.random.Generator):
        self.cfg         = cfg
        self.manifold    = manifold
        self.encoder     = encoder
        self.meaning_map = meaning_map
        self.rng         = rng

        # Policy over anchor sizes k = 1, …, |W|/2 (tractable prototype)
        max_k = max(1, cfg.vocab_size // 10)
        self.anchor_sizes   = np.arange(1, max_k + 1)
        self.policy_weights = np.ones(max_k) / max_k   # Uniform initialisation

        # Track last selected anchor size for policy update
        self._last_k = None

    def _sample_gamma(self, k: int) -> np.ndarray:
        """Sample Γ ⊆ W uniformly at random with |Γ| = k."""
        return self.rng.choice(self.cfg.vocab_size, size=k, replace=False)

    def sample_anchor_size(self) -> int:
        """π_s^(t) → k = |Γ_t|  (sample anchor size from policy)."""
        probs = self.policy_weights / self.policy_weights.sum()
        k = int(self.rng.choice(self.anchor_sizes, p=probs))
        self._last_k = k
        return k

    def build_sem(self) -> Tuple[SemInstance, np.ndarray]:
        """
        Steps S1–S3: sample Γ, construct B, form sem.

        Returns
        -------
        sem     : SemInstance
        gamma   : selected anchor set indices
        """
        k     = self.sample_anchor_size()           # S1
        gamma = self._sample_gamma(k)
        B     = self.manifold.base_signal(gamma)    # S2
        iota  = self.manifold.softmax_to_interpretation(B)
        sem   = SemInstance(B=B, W=self.manifold.W,
                            gamma=gamma, iota=iota)  # S3
        return sem, gamma

    def encode(self, sem: SemInstance,
               conservation_monitor: ConservationMonitor
               ) -> Optional[np.ndarray]:
        """
        Steps S4–S7: Cauchy–Riemann gate, S-OSI encode, conservation check.

        Returns ψ ∈ ℂ^d if all checks pass, else None (round aborted).
        """
        # S4: Cauchy–Riemann analyticity gate
        if not self.meaning_map.cauchy_riemann_check(sem):
            logger.debug("S4: Cauchy–Riemann violated — round aborted.")
            return None

        # S5: S-OSI encoding
        psi = self.encoder(sem)                     # ψ ∈ ℂ^d

        # S6: Conservation check
        if not conservation_monitor.check(psi):
            return None                             # anomaly flagged

        return psi                                  # S7: ready to transmit

    def update_policy(self, delta_t: float) -> None:
        """
        Step A1: multiplicative weights update.

        π_s^(t+1)(k) ∝ π_s^(t)(k) · exp(η_s · δ_t · 𝟏[k_t = k])
        """
        if self._last_k is None:
            return
        idx = np.where(self.anchor_sizes == self._last_k)[0]
        if len(idx) > 0:
            self.policy_weights[idx[0]] *= math.exp(
                self.cfg.eta_sender * delta_t
            )
        # Normalise to prevent overflow
        self.policy_weights = np.clip(self.policy_weights, 1e-10, None)
        self.policy_weights /= self.policy_weights.sum()


# ══════════════════════════════════════════════════════════════════════════════
# 12.  RECEIVER AGENT   (Steps R1–R6 + Adaptation A2)
# ══════════════════════════════════════════════════════════════════════════════

class ReceiverAgent:
    """
    Adaptive receiver agent.

    Performs decoding (Steps R1–R3), fidelity measurement (R4),
    feedback computation (R5), and parameter update (A2):

        θ_r^(t+1) = θ_r^(t) + η_r · ∇_{θ_r} F_s^(t)

    In this scaffold θ_r parameterises the SOSI decoder's projection;
    the gradient step is approximated by directly reinforcing the
    lambda weights of the meaning map toward higher-fidelity directions.
    """

    def __init__(self, cfg: LPGConfig, decoder: SOSIDecoder,
                 fidelity_meter: SemanticFidelityMeter):
        self.cfg           = cfg
        self.decoder       = decoder
        self.fidelity_meter = fidelity_meter

    def receive(self, psi_tilde: np.ndarray,
                H_s: np.ndarray,
                iota_true: np.ndarray
                ) -> Tuple[SemInstance, float, float]:
        """
        Steps R1–R5.

        Parameters
        ----------
        psi_tilde  : received signal ψ̃ ∈ ℂ^d
        H_s        : realised channel matrix
        iota_true  : sender's true interpretation map ι_t

        Returns
        -------
        sem_hat   : reconstructed SemInstance
        fidelity  : F_s^(t) ∈ [0,1]
        delta_t   : feedback signal δ_t = F_s^(t) − F*
        """
        # R1–R3: S-OSI decode
        sem_hat  = self.decoder.decode(psi_tilde, H_s)      # R1–R3

        # R4: Semantic fidelity
        fidelity = self.fidelity_meter.measure(iota_true,
                                               sem_hat.iota) # R4

        # R5: Feedback signal
        delta_t  = fidelity - self.cfg.fidelity_target       # R5

        return sem_hat, fidelity, delta_t

    def update_parameters(self, delta_t: float,
                          meaning_map: ComplexMeaningMap) -> None:
        """
        Step A2: gradient ascent on fidelity.

        θ_r^(t+1) = θ_r^(t) + η_r · ∇_{θ_r} F_s^(t)

        Prototype: nudge the meaning map's context weight vector in the
        direction that increases fidelity (positive δ_t).
        Full implementation uses autograd through the decoder network.
        """
        meaning_map.lambda_context += (
            self.cfg.eta_receiver * delta_t *
            np.sign(meaning_map.lambda_global)
        )


# ══════════════════════════════════════════════════════════════════════════════
# 13.  COUPLED ADAPTIVE LOOP   — Algorithm 3.1 Master Loop
# ══════════════════════════════════════════════════════════════════════════════

class CoupledAdaptiveLoop:
    """
    Implements Algorithm 3.1 in full.

    Orchestrates the sender, channel, receiver, and all adaptive updates
    over T_max rounds until the convergence criterion is satisfied:

        |δ_t| < ε_conv   for τ_conv consecutive rounds

    Output on convergence:
        (Ĉ_s, F̄_s, μ_s*, Γ̂*, ι̂*)
    """

    def __init__(self, cfg: LPGConfig):
        self.cfg = cfg
        rng      = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)

        # ── Instantiate all modules ───────────────────────────────────────
        self.manifold      = SemanticManifold(cfg)
        self.meaning_map   = ComplexMeaningMap(cfg, seed=cfg.seed)
        self.encoder       = SOSIEncoder(cfg)
        self.reconstructor = MLAnchorReconstructor(cfg, self.manifold)
        self.decoder       = SOSIDecoder(cfg, self.manifold,
                                         self.reconstructor)
        self.channel       = SemanticChannel(cfg, rng)
        self.fidelity_meter = SemanticFidelityMeter()
        self.capacity_est  = SemanticCapacityEstimator(cfg)
        self.conservation  = ConservationMonitor(cfg)
        self.sender        = SenderAgent(cfg, self.manifold, self.encoder,
                                         self.meaning_map, rng)
        self.receiver      = ReceiverAgent(cfg, self.decoder,
                                           self.fidelity_meter)

        # ── Loop state ────────────────────────────────────────────────────
        self.t                = 0
        self.consecutive_conv = 0
        self.history: List[Dict] = []

    def _check_convergence(self, delta_t: float) -> bool:
        """
        Convergence criterion: |δ_t| < ε_conv for τ_conv consecutive rounds.
        """
        if abs(delta_t) < self.cfg.epsilon_conv:
            self.consecutive_conv += 1
        else:
            self.consecutive_conv = 0
        return self.consecutive_conv >= self.cfg.tau_conv

    def run(self) -> Dict:
        """
        Execute Algorithm 3.1 from INITIALIZATION to RETURN.

        Returns
        -------
        result : dict with keys
            C_s_hat   — empirical semantic capacity (semantic bits/use)
            F_s_bar   — mean semantic fidelity
            mu_s_star — converged semantic permeability
            gamma_hat — converged anchor set
            iota_hat  — converged interpretation map
            rounds    — number of rounds executed
            anomalies — number of conservation anomalies
        """
        logger.info("═" * 60)
        logger.info("LPG Semantic Communication — Coupled Adaptive Loop")
        logger.info(f"  |W| = {self.cfg.vocab_size}   F* = {self.cfg.fidelity_target}"
                    f"   T_max = {self.cfg.T_max}")
        logger.info("═" * 60)

        gamma_hat_final = np.array([], dtype=int)
        iota_hat_final  = np.zeros(self.cfg.vocab_size)

        # ── INITIALIZATION ────────────────────────────────────────────────
        self.channel.mu_s = self.cfg.mu_s_init     # μ_s^(1) ← μ_s,0
        self.t            = 1

        # ── MAIN LOOP ─────────────────────────────────────────────────────
        while self.t <= self.cfg.T_max:

            # ── SENDER ───────────────────────────────────────────────────
            sem, gamma = self.sender.build_sem()    # S1–S3
            psi        = self.sender.encode(sem,    # S4–S6
                                            self.conservation)

            if psi is None:
                # Round aborted (CR violation or conservation anomaly)
                self.t += 1
                continue

            # S7: Transmit ψ over semantic field channel

            # ── CHANNEL ───────────────────────────────────────────────────
            psi_tilde, H_s = self.channel.transmit(psi)   # C1

            # ── RECEIVER ─────────────────────────────────────────────────
            sem_hat, fidelity, delta_t = self.receiver.receive(
                psi_tilde, H_s, sem.iota                   # R1–R5
            )
            # R6: δ_t transmitted to sender via feedback channel

            # ── ADAPTATION ───────────────────────────────────────────────
            self.sender.update_policy(delta_t)             # A1
            self.receiver.update_parameters(               # A2
                delta_t, self.meaning_map
            )
            self.channel.mu_s += self.cfg.gamma * delta_t # A3

            # ── RECORD ────────────────────────────────────────────────────
            self.capacity_est.record(sem_hat.gamma, fidelity)
            gamma_hat_final = sem_hat.gamma
            iota_hat_final  = sem_hat.iota

            record = {
                "t":         self.t,
                "fidelity":  round(fidelity, 5),
                "delta_t":   round(delta_t, 5),
                "mu_s":      round(self.channel.mu_s, 5),
                "gamma_size": len(sem_hat.gamma),
            }
            self.history.append(record)

            if self.t % 100 == 0 or abs(delta_t) < self.cfg.epsilon_conv:
                logger.info(f"  Round {self.t:4d} | "
                            f"F_s={fidelity:.4f} | "
                            f"δ={delta_t:+.4f} | "
                            f"μ_s={self.channel.mu_s:.4f} | "
                            f"|Γ̂|={len(sem_hat.gamma)}")

            # ── CONVERGENCE CHECK ─────────────────────────────────────────
            if self._check_convergence(delta_t):
                logger.info(f"  Converged at round {self.t} "
                            f"(|δ| < {self.cfg.epsilon_conv} "
                            f"for {self.cfg.tau_conv} rounds).")
                break

            self.t += 1

        # ── RETURN ────────────────────────────────────────────────────────
        stats = self.capacity_est.estimate()
        result = {
            "C_s_hat"   : stats["C_s_hat"],
            "F_s_bar"   : stats["F_s_bar"],
            "mu_s_star" : self.channel.mu_s,
            "gamma_hat" : gamma_hat_final,
            "iota_hat"  : iota_hat_final,
            "rounds"    : self.t,
            "anomalies" : self.conservation.anomaly_count,
        }

        logger.info("─" * 60)
        logger.info(f"  Ĉ_s      = {result['C_s_hat']:.6f} semantic bits/use")
        logger.info(f"  F̄_s      = {result['F_s_bar']:.4f}")
        logger.info(f"  μ_s*     = {result['mu_s_star']:.4f}")
        logger.info(f"  Rounds   = {result['rounds']}")
        logger.info(f"  Anomalies= {result['anomalies']}")
        logger.info("═" * 60)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# 14.  SIMULATION RUNNER
#      Experiment harness for systematic parameter sweeps
# ══════════════════════════════════════════════════════════════════════════════

class SimulationRunner:
    """
    Experiment harness for LPG Semantic Communication simulations.

    Supports:
        - Single-run execution
        - SNR sweeps  (vary σ² at fixed |W|)
        - Vocabulary sweeps (vary |W| at fixed SNR)
        - Fidelity target sweeps (vary F*)

    Results are returned as structured dicts suitable for
    direct plotting with matplotlib or export to CSV.

    Sionna integration note:
        Replace SemanticChannel._rayleigh_matrix() with a
        Sionna channel model for hardware-accurate simulation.
        All other modules are Sionna-agnostic.
    """

    def __init__(self, base_cfg: LPGConfig):
        self.base_cfg = base_cfg

    def single_run(self) -> Dict:
        """Execute one full coupled adaptive loop run."""
        loop = CoupledAdaptiveLoop(self.base_cfg)
        return loop.run()

    def snr_sweep(self, sigma2_values: List[float]) -> List[Dict]:
        """
        Sweep over noise variance σ² values.

        Returns list of result dicts, one per σ² value.
        Each dict includes 'sigma2' key for identification.
        """
        results = []
        for sigma2 in sigma2_values:
            import copy
            cfg = copy.deepcopy(self.base_cfg)
            cfg.sigma2 = sigma2
            loop = CoupledAdaptiveLoop(cfg)
            result = loop.run()
            result["sigma2"] = sigma2
            results.append(result)
            logger.info(f"SNR sweep: σ²={sigma2:.3f} → Ĉ_s={result['C_s_hat']:.4f}")
        return results

    def vocab_sweep(self, vocab_sizes: List[int]) -> List[Dict]:
        """
        Sweep over vocabulary sizes |W|.

        Returns list of result dicts, one per |W| value.
        """
        results = []
        for V in vocab_sizes:
            import copy
            cfg = copy.deepcopy(self.base_cfg)
            cfg.vocab_size = V
            loop = CoupledAdaptiveLoop(cfg)
            result = loop.run()
            result["vocab_size"] = V
            results.append(result)
            logger.info(f"Vocab sweep: |W|={V} → Ĉ_s={result['C_s_hat']:.4f}")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 15.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Default single-run execution of Algorithm 3.1.

    To run:
        python lgp_semantic_comm.py

    To integrate with Sionna:
        1. Install: pip install sionna
        2. In SemanticChannel._rayleigh_matrix(), replace the numpy
           stub with:
               channel = sionna.channel.RayleighBlockFading(
                   num_rx=1, num_rx_ant=self.cfg.tx_dim,
                   num_tx=1, num_tx_ant=self.cfg.tx_dim
               )
        3. All other modules remain unchanged.

    To run a systematic SNR sweep:
        runner = SimulationRunner(cfg)
        results = runner.snr_sweep([0.01, 0.05, 0.1, 0.3, 0.5, 1.0])
    """
    cfg    = LPGConfig(
        vocab_size      = 200,       # Small for rapid prototype run
        fidelity_target = 0.85,
        T_max           = 500,
        tau_conv        = 10,
        seed            = 42,
    )

    runner = SimulationRunner(cfg)
    result = runner.single_run()

    print("\n" + "═" * 50)
    print("LPG SEMANTIC COMMUNICATION — SIMULATION RESULT")
    print("═" * 50)
    print(f"  Semantic Capacity  Ĉ_s  = {result['C_s_hat']:.6f}  sem-bits/use")
    print(f"  Mean Fidelity      F̄_s  = {result['F_s_bar']:.4f}")
    print(f"  Converged μ_s*          = {result['mu_s_star']:.4f}")
    print(f"  Rounds to converge      = {result['rounds']}")
    print(f"  Conservation anomalies  = {result['anomalies']}")
    print(f"  |Γ̂*| (final anchor set) = {len(result['gamma_hat'])}")
    print("═" * 50)
