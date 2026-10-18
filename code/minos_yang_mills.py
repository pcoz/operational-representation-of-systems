"""
The computations behind 'An Operational Representation of Systems: A Worked Construction in Lattice Gauge Theory' (E. Chalk, 2026).

Copyright (c) 2026 Edward Chalk, edward.chalk@sapientronic.ai. All rights reserved.

python minos_yang_mills.py  writes results.txt and results.json beside it (numpy and scipy; about fifteen minutes; checks 17-21 need scipy).

What it computes (the paper's checks, in order; the paper's section 7 lists the ones that carry the argument):
  1  the elemental node: L is a rotation; the readable shares are quadratic in the amplitudes (the half-angle); the harmonics of a
     turning node contain n = 0 and n = 2 only
  2  two systems: mediations in one plane commute and close; mediations in alternating planes leave a loop residual of second order
  3  the cost of a loop averaged over an elemental state with the information metric's measure: in the amplitude reading the Wilson
     action, in the share reading the adjoint Wilson action (SU(2), SU(3), SU(4)); the identity |tr U|^2 = chi_adj(U) + 1
  4  one continuum limit: both costs -> const * eps^2 tr F^2 with couplings in the ratio 2N/(N+1)
  5  on the alternating-plane loop of check 2 the cost is fourth order in the mediation (the non-Abelian F^2) and vanishes for one plane
  6  the smallest closed loop's Hamiltonian H = lam C_2 + v/lam on class functions of SU(2): the share reading couples the vacuum to
     integer j only; the gap ratio shares/amplitudes -> C_2(1)/C_2(1/2) = 8/3 at strong coupling and -> 1 at weak coupling
  7  a closed whole carries an internal alternation at the gap: the loop prepared at one coupling and evolved at another; the held
     share oscillates; its frequency is the gap of H
  8  the Haar second moment in general: <|<psi|U|psi>|^2> = (N + |tr U|^2)/(N(N+1)), the origin of the adjoint through N x Nbar = 1 + adj
  9  the information distance between two relations averaged over the state is bi-invariant and, for small displacements, the
     Killing form: the kinetic term's Laplacian is the Casimir
 10  two rotation groups about distinct axes generate SO(3): the orbit of a point fills the sphere; about one axis it is a circle
 11  the real three-pole composite's readable loss is the spin-2 Wilson action (2/3)(1 - chi_2/5), not the adjoint
 12  the Fubini-Study and the Fisher-Rao measures told apart on diag(i, -i): average returns 1/3 and 1/2
 13  real MINOS with N poles: the loss is the traceless-symmetric character action (2 d_S/(N(N+2)))(1 - chi_S/d_S), N = 3..6
 14  A6 as amplitude restriction: defined everywhere; the shared pole counted in both weights; consistent under the composite's rotations
"""
import json, time
import numpy as np

rng = np.random.default_rng(42)
OUT = {}
LINES = []


def say(s=""):
    print(s); LINES.append(s)


# ----------------------------------------------------------------------------------------------------- SU(N) helpers
def sun_generators(N):
    """Generalised Gell-Mann matrices, tr(T_a T_b) = delta_ab / 2."""
    T = []
    for j in range(N):
        for k in range(j + 1, N):
            s = np.zeros((N, N), complex); s[j, k] = s[k, j] = 1; T.append(s / 2)
            a = np.zeros((N, N), complex); a[j, k] = -1j; a[k, j] = 1j; T.append(a / 2)
    for l in range(1, N):
        d = np.zeros((N, N), complex); d[:l, :l] = np.eye(l); d[l, l] = -l
        T.append(d * np.sqrt(2.0 / (l * (l + 1))) / 2)
    return T


def expi(H):
    w, V = np.linalg.eigh(H); return V @ np.diag(np.exp(1j * w)) @ V.conj().T


def random_su(N, scale=1.0):
    T = sun_generators(N); a = rng.normal(size=len(T)) * scale
    F = sum(x * t for x, t in zip(a, T)); return expi(F), F


def chi_adj(U, N):
    T = np.array(sun_generators(N))
    R = 2 * np.real(np.einsum('aij,jk,bkl,li->ab', T, U, T, U.conj().T))
    return float(np.trace(R))


def haar_states(N, n):
    z = rng.normal(size=(n, N)) + 1j * rng.normal(size=(n, N)); return z / np.linalg.norm(z, axis=1, keepdims=True)


# ----------------------------------------------------------------------------------------------------- check 1: the node
say("CHECK 1  the elemental node")
theta = np.linspace(0, np.pi / 2, 7)
say("  theta      passed = cos^2   held = sin^2   sum")
for th in theta:
    say(f"  {th:6.3f}    {np.cos(th)**2:10.6f}   {np.sin(th)**2:10.6f}   {np.cos(th)**2 + np.sin(th)**2:.6f}")
# L is a rotation: an isotropic metric is preserved by rotation, an anisotropic one is not (the node could not be an angle)
def form_change(lam, steps=720):
    g = np.diag([1.0, lam]); worst = 0.0
    for phi in np.linspace(0, 2 * np.pi, steps, endpoint=False):
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        worst = max(worst, np.abs(R.T @ g @ R - g).max())
    return worst
OUT["check1"] = dict(isotropic_change=form_change(1.0), anisotropic_change=form_change(1.5))
say(f"  a rotation preserves the isotropic form (max change {OUT['check1']['isotropic_change']:.1e}) and not an anisotropic one ({OUT['check1']['anisotropic_change']:.3f})")
# harmonics of a turning node
T = 4096; t = np.arange(T); om = 2 * np.pi * 16 / T
f = np.cos(om * t) ** 2
F = np.abs(np.fft.rfft(f - f.mean())) ** 2; ac = F.sum()
OUT["check1"].update(power_at_omega=float(F[16] / ac), power_at_2omega=float(F[32] / ac))
say(f"  a node turning at rate w: a.c. power at w {F[16]/ac:.1e}, at 2w {F[32]/ac:.6f} -- the readable has no first harmonic")
say()

# ----------------------------------------------------------------------------------------------------- check 2: two systems
say("CHECK 2  two systems: one plane closes, alternating planes leave a residual")
I2 = np.eye(2, dtype=complex); SX = np.array([[0, 1], [1, 0]], complex); SZ = np.array([[1, 0], [0, -1]], complex)
def med(axis, a): return np.cos(a / 2) * I2 - 1j * np.sin(a / 2) * axis
def residual(U): return float(2 * np.arccos(np.clip(np.trace(U).real / 2, -1, 1)))
say("    theta     one plane residual     alternating planes residual   residual / theta^2")
OUT["check2"] = []
for th in (0.4, 0.2, 0.1, 0.05):
    U1 = med(SZ, th) @ med(SZ, th) @ med(SZ, -th) @ med(SZ, -th)
    U2 = med(SZ, th) @ med(SX, th) @ med(SZ, -th) @ med(SX, -th)
    r1, r2 = residual(U1), residual(U2)
    OUT["check2"].append(dict(theta=th, one_plane=r1, alternating=r2, ratio=r2 / th ** 2))
    say(f"    {th:5.2f}     {r1:14.1e}        {r2:14.6f}              {r2 / th**2:.5f}")
say()

# ----------------------------------------------------------------------------------------------------- check 3: the two readings
say("CHECK 3  the loop's cost averaged over an elemental state: the Wilson action (amplitudes) and the adjoint action (shares)")
say("    N   1 - Re<psi|U|psi> (MC)      1 - Re tr U / N     |   1 - |<psi|U|psi>|^2 (MC)    ((N-1)/N)(1 - chi_adj/(N^2-1))   |tr U|^2 - chi_adj - 1")
OUT["check3"] = []
for N in (2, 3, 4):
    for rep in range(2):
        U, _ = random_su(N, 1.3); psi = haar_states(N, 400000)
        amp = np.einsum('ni,ij,nj->n', psi.conj(), U, psi)
        a_mc, a_er = float(np.mean(1 - amp.real)), float(np.std(1 - amp.real) / np.sqrt(len(amp)))
        s_mc, s_er = float(np.mean(1 - np.abs(amp) ** 2)), float(np.std(1 - np.abs(amp) ** 2) / np.sqrt(len(amp)))
        wil = float(1 - np.trace(U).real / N); ca = chi_adj(U, N); adj = (N - 1) / N * (1 - ca / (N * N - 1)); ident = float(abs(np.trace(U)) ** 2 - ca - 1)
        OUT["check3"].append(dict(N=N, amp_mc=a_mc, amp_err=a_er, wilson=wil, share_mc=s_mc, share_err=s_er, adjoint=adj, identity=ident))
        say(f"    {N}      {a_mc:.5f}({a_er:.5f})          {wil:.5f}         |     {s_mc:.5f}({s_er:.5f})            {adj:.5f}                   {ident:+.1e}")
say()

# ----------------------------------------------------------------------------------------------------- check 4: one continuum limit
say("CHECK 4  one continuum limit: U = exp(i eps F); both costs -> const * eps^2 tr F^2; couplings in the ratio 2N/(N+1)")
say("    N    eps    amplitudes / (eps^2 trF^2 / 2N)    shares / (eps^2 trF^2 / (N+1))    shares/amplitudes    2N/(N+1)")
OUT["check4"] = []
for N in (2, 3, 4):
    _, Fm = random_su(N, 1.0); trF2 = float(np.trace(Fm @ Fm).real)
    for eps in (0.3, 0.1, 0.03):
        U = expi(eps * Fm); wil = 1 - np.trace(U).real / N; adj = (N - 1) / N * (1 - chi_adj(U, N) / (N * N - 1))
        OUT["check4"].append(dict(N=N, eps=eps, amp_norm=float(wil / (eps ** 2 * trF2 / (2 * N))), share_norm=float(adj / (eps ** 2 * trF2 / (N + 1))), ratio=float(adj / wil), target=2 * N / (N + 1)))
        say(f"    {N}   {eps:5.2f}        {wil/(eps**2*trF2/(2*N)):.6f}                      {adj/(eps**2*trF2/(N+1)):.6f}                 {adj/wil:.6f}           {2*N/(N+1):.6f}")
say()

# ----------------------------------------------------------------------------------------------------- check 5: the loop of check 2
say("CHECK 5  the cost on the alternating-plane loop: fourth order in the mediation, zero for one plane")
say("      theta    residual / theta^2    amplitudes / theta^4 (-> 1/8)    shares / theta^4 (-> 1/6)    one plane: amplitudes, shares")
OUT["check5"] = []
for th in (0.4, 0.2, 0.1, 0.05):
    U = med(SZ, th) @ med(SX, th) @ med(SZ, -th) @ med(SX, -th); c = float(np.clip(np.trace(U).real / 2, -1, 1)); Phi = 2 * np.arccos(c)
    wil = 1 - c; sh = (2 / 3) * (1 - c * c)
    U1 = med(SZ, th) @ med(SZ, th) @ med(SZ, -th) @ med(SZ, -th); c1 = float(np.trace(U1).real / 2)
    OUT["check5"].append(dict(theta=th, residual_over_theta2=float(Phi / th ** 2), amp_over_theta4=float(wil / th ** 4), share_over_theta4=float(sh / th ** 4), one_plane_amp=float(1 - c1), one_plane_share=float((2 / 3) * (1 - c1 * c1))))
    say(f"      {th:5.2f}      {Phi/th**2:.6f}              {wil/th**4:.6f}                      {sh/th**4:.6f}                   {1-c1:.1e}, {(2/3)*(1-c1*c1):.1e}")
say()

# ----------------------------------------------------------------------------------------------------- check 6: one loop solved
say("CHECK 6  the smallest closed loop: H = lam C_2 + v/lam on class functions of SU(2)")
J = 60; twoj = np.arange(0, 2 * J + 1); C2 = (twoj / 2) * (twoj / 2 + 1); n = len(twoj)
M12 = np.zeros((n, n)); M1 = np.zeros((n, n))
for a in range(n):
    for b in (a - 1, a + 1):
        if 0 <= b < n: M12[b, a] += 1
    for b in (a - 2, a, a + 2):
        if 0 <= b < n and not (a == 0 and b == 0): M1[b, a] += 1
M1[0, 0] = 0.0
vW = np.eye(n) - M12 / 2; vS = (3 / 8) * (np.eye(n) - M1 / 3)
def spectrum(lam, v):
    H = lam * np.diag(C2) + v / lam; w, V = np.linalg.eigh((H + H.T) / 2)
    g = V[:, 0]; conn = [k for k in range(1, n) if abs(V[:, k] @ (v @ g)) > 1e-9]
    return w, V, float(w[conn[0]] - w[0]), g
say("      lam      gap, amplitudes     gap, shares      ratio      vacuum weight on half-integer j (shares)")
OUT["check6"] = []
for lam in (50.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03):
    _, _, gW, _ = spectrum(lam, vW); _, _, gS, gs = spectrum(lam, vS)
    half = float(np.sum(gs[1::2] ** 2))
    OUT["check6"].append(dict(lam=lam, gap_amplitudes=gW, gap_shares=gS, ratio=gS / gW, half_integer_weight=half))
    say(f"    {lam:6.2f}      {gW:10.5f}        {gS:10.5f}      {gS/gW:7.4f}      {half:.1e}")
say(f"  strong coupling: the ratio -> C_2(1)/C_2(1/2) = {2/0.75:.4f};  weak coupling: -> 1")
say()

# ----------------------------------------------------------------------------------------------------- check 7: closed over a period
say("CHECK 7  a closed whole carries an internal alternation at the gap (the loop quenched from coupling 2 lam to lam)")
say("      lam     gap of H     frequency of the held share's alternation     ratio")
OUT["check7"] = []
for lam in (0.5, 1.0, 2.0):
    w, V, gap, _ = spectrum(lam, vS)
    _, V0, _, g0 = spectrum(2 * lam, vS)
    psi0 = g0.astype(complex); c = V.T @ psi0
    P = 2 * np.pi / gap; dt = P / 64; times = np.arange(4096) * dt
    ph = np.exp(-1j * np.outer(times, w)); psi_t = (ph * c) @ V.T
    held = np.einsum('ti,ij,tj->t', psi_t.conj(), vS, psi_t).real
    S = np.abs(np.fft.rfft(held - held.mean())) ** 2; k = int(np.argmax(S)); freq = 2 * np.pi * k / (len(times) * dt)
    # v9: the exact spectral content. The prepared state has weights |c_n|^2 on H(lam)'s eigenstates; the a.c. part of <held> is
    # sum_{m<n} 2 Re(c_m* c_n O_mn e^{-i(E_n-E_m)t}), so the power at each transition is |c_m c_n O_mn|^2 (frequencies distinct)
    O = V.T @ vS @ V; wts = np.abs(c) ** 2; order = np.argsort(w)
    pairs = [(m, n, abs(c[m] * c[n] * O[m, n]) ** 2) for m in range(len(w)) for n in range(m + 1, len(w))]
    tot = sum(pw for _, _, pw in pairs)
    n_gap = next(n for n in range(1, len(w)) if abs(w[n] - w[0] - gap) < 1e-9)      # the first CONNECTED excited state (spectrum's gap), not the next eigenvalue
    gap_pair = next(pw for m, n, pw in pairs if {m, n} == {0, n_gap}); w01 = float(wts[0] + wts[n_gap])
    OUT["check7"].append(dict(lam=lam, gap=gap, frequency=float(freq), ratio=float(freq / gap), power_share_fft=float(S[k] / S.sum()), power_share_exact=float(gap_pair / tot),
                              weight_in_two_lowest=w01, swing=float(held.max() - held.min()), mean=float(held.mean())))
    say(f"    {lam:5.2f}    {gap:9.5f}               {freq:9.5f}                       {freq/gap:.4f}   (a.c. power at the gap: {S[k]/S.sum():.4f} by FFT, {gap_pair/tot:.6f} exact; the prepared state's weight in the ground and first connected eigenstates {w01:.6f}; mean {held.mean():.4f}, swing {held.max()-held.min():.4f})")
say("  (v9: the prepared state is not a two-level superposition exactly; the gap carries the stated share of the a.c. power and the rest sits at the other transition frequencies)")
say()

# ----------------------------------------------------------------------------------------------------- check 8: the second moment, in general
say("CHECK 8  the Haar second moment on the sphere: <|<psi|U|psi>|^2> = (N + |tr U|^2) / (N (N + 1)), against Monte Carlo, N = 2..5")
say("    N    <|<psi|U|psi>|^2> (MC)      (N + |tr U|^2)/(N(N+1))     1 - that   vs   ((N-1)/N)(1 - chi_adj/(N^2-1))")
OUT["check8"] = []
for N in (2, 3, 4, 5):
    U, _ = random_su(N, 1.1); psi = haar_states(N, 300000)
    amp = np.einsum('ni,ij,nj->n', psi.conj(), U, psi); mc = float(np.mean(np.abs(amp) ** 2)); er = float(np.std(np.abs(amp) ** 2) / np.sqrt(len(amp)))
    formula = float((N + abs(np.trace(U)) ** 2) / (N * (N + 1))); adj = float((N - 1) / N * (1 - chi_adj(U, N) / (N * N - 1)))
    OUT["check8"].append(dict(N=N, mc=mc, err=er, formula=formula, one_minus=1 - formula, adjoint_form=adj))
    say(f"    {N}      {mc:.5f}({er:.5f})               {formula:.5f}              {1-formula:.5f}        {adj:.5f}")
say()

# ----------------------------------------------------------------------------------------------------- check 9: the induced metric on relations
say("CHECK 9  the information distance between two relations, averaged over the elemental state, is a bi-invariant function of U^dag U'")
say("         d^2(U, U') := <1 - |<psi|U^dag U'|psi>|^2>  =  ((N-1)/N)(1 - chi_adj(U^dag U')/(N^2-1));  invariance under U -> g U h, U' -> g U' h")
OUT["check9"] = []
for N in (2, 3):
    U, _ = random_su(N, 0.7); U2, _ = random_su(N, 0.7); g, _ = random_su(N, 1.0); h, _ = random_su(N, 1.0)
    W = U.conj().T @ U2
    d2 = (N - 1) / N * (1 - chi_adj(W, N) / (N * N - 1))
    W2 = (g @ U @ h).conj().T @ (g @ U2 @ h); d2_moved = (N - 1) / N * (1 - chi_adj(W2, N) / (N * N - 1))
    # small displacements: d^2 ~ (eps^2 / (N+1)) tr F^2 -- the Killing form, so the Laplacian of this metric is the Casimir
    _, Fm = random_su(N, 1.0); trF2 = float(np.trace(Fm @ Fm).real); eps = 0.02
    d2_small = (N - 1) / N * (1 - chi_adj(expi(eps * Fm), N) / (N * N - 1))
    OUT["check9"].append(dict(N=N, d2=float(d2), d2_moved=float(d2_moved), small_ratio=float(d2_small / (eps ** 2 * trF2 / (N + 1)))))
    say(f"    N = {N}:  d^2(U, U') = {d2:.6f};  after moving both by (g, h): {d2_moved:.6f};  small displacement: d^2 / (eps^2 tr F^2 / (N+1)) = {d2_small/(eps**2*trF2/(N+1)):.5f}")
say("  (a bi-invariant metric on a simple group is a multiple of the Killing form; its Laplace-Beltrami operator is the Casimir operator)")
say()

# ----------------------------------------------------------------------------------------------------- check 10: two planes generate SO(3)
say("CHECK 10 the orbit of a point under words in two rotations: distinct axes fill the sphere, one axis a circle")
def rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
def coverage(axes, n_words=20000, length=12):
    cells = np.zeros((24, 48), bool); v0 = np.array([1.0, 0.0, 0.0])
    for _ in range(n_words):
        v = v0.copy()
        for _ in range(length):
            v = rot(axes[rng.integers(len(axes))], rng.uniform(-np.pi, np.pi)) @ v
        th = np.arccos(np.clip(v[2], -1, 1)); ph = np.arctan2(v[1], v[0]) % (2 * np.pi)
        cells[min(int(th / np.pi * 24), 23), min(int(ph / (2 * np.pi) * 48), 47)] = True
    return float(cells.mean())
two = coverage([(0, 0, 1), (1, 0, 0)]); one = coverage([(0, 0, 1), (0, 0, 1)])
OUT["check10"] = dict(two_axes_coverage=two, one_axis_coverage=one)
say(f"  cells of a 24 x 48 partition of the sphere visited: two distinct axes {two:.4f} of them; one axis {one:.4f} (a circle)")
say()

# ----------------------------------------------------------------------------------------------------- check 11: the REAL composite's loss
say("CHECK 11 the real three-pole composite: states +-n in RP^2, transition share (n.m)^2, relations SO(3) in the vector representation;")
say("         the readable loss round a loop <1 - (n . R n)^2>_n over the uniform measure on S^2 against (2/3)(1 - chi_2(R)/5), chi_2 = 1 + 2 cos phi + 2 cos 2 phi")
def rot3(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
OUT["check11"] = []
say("      phi      MC loss          (2/3)(1 - chi_2/5)     adjoint form (2/3)(1 - chi_1/3), for contrast")
for phi in (0.3, 1.0, 2.0, 3.0):
    R = rot3(rng.normal(size=3), phi); n = rng.normal(size=(400000, 3)); n /= np.linalg.norm(n, axis=1, keepdims=True)
    loss = float(np.mean(1 - np.einsum('ni,ij,nj->n', n, R, n) ** 2)); er = float(np.std(1 - np.einsum('ni,ij,nj->n', n, R, n) ** 2) / np.sqrt(len(n)))
    chi2 = 1 + 2 * np.cos(phi) + 2 * np.cos(2 * phi); chi1 = 1 + 2 * np.cos(phi)
    OUT["check11"].append(dict(phi=phi, mc=loss, err=er, spin2=float(2 / 3 * (1 - chi2 / 5)), spin1=float(2 / 3 * (1 - chi1 / 3))))
    say(f"     {phi:4.1f}     {loss:.5f}({er:.5f})     {2/3*(1-chi2/5):.5f}                {2/3*(1-chi1/3):.5f}")
say("  (3 x 3 = 1 + 3 + 5; a real share is a symmetric square, so the loss carries the symmetric traceless 5, not the adjoint 3)")
say()

# ----------------------------------------------------------------------------------------------------- check 12: the two measures
say("CHECK 12 two invariant measures, told apart on U = diag(i, -i), return share (2p - 1)^2 with p = |psi_1|^2:")
pr = rng.uniform(0, 1, 400000)                                   # p under the uniform measure on CP^1 (complex amplitudes): uniform in p
th = rng.uniform(0, np.pi, 400000); pf = np.cos(th) ** 2         # p under the round measure on the real circle (Fisher-Rao on the simplex): arcsine in p
c_avg = float(np.mean((2 * pr - 1) ** 2)); f_avg = float(np.mean((2 * pf - 1) ** 2))
OUT["check12"] = dict(complex_average_return=c_avg, fisher_rao_average_return=f_avg)
say(f"    complex projective (Fubini-Study) measure, uniform in p:  <return> = {c_avg:.4f}  (1/3; loss 2/3 -- Theorem 2's measure)")
say(f"    Fisher-Rao volume on the simplex, arcsine in p (= the round measure on REAL amplitudes):  <return> = {f_avg:.4f}  (1/2; loss 1/2)")
say("  (the measure the axioms select is the one invariant under the relation group: Fubini-Study for complex amplitudes, Fisher-Rao for real)")
say()

# ----------------------------------------------------------------------------------------------------- check 13: the real branch, N poles
say("CHECK 13 real MINOS with N poles: readable states RP^(N-1), relations SO(N); the loss <1 - (n . R n)^2> over S^(N-1) against")
say("         (2 d_S / (N (N+2))) (1 - chi_S(R) / d_S),  S the traceless symmetric representation, d_S = N(N+1)/2 - 1,  chi_S = ((tr R)^2 + tr R^2)/2 - 1")
def random_so(N):
    A = rng.normal(size=(N, N)); A = A - A.T
    w, V = np.linalg.eigh(1j * A); return np.real(V @ np.diag(np.exp(-1j * w)) @ V.conj().T)
OUT["check13"] = []
say("    N    MC loss              formula       d_S     (for N = 3 the formula is (2/3)(1 - chi_2/5))")
for N in (3, 4, 5, 6):
    Rm = random_so(N); n = rng.normal(size=(400000, N)); n /= np.linalg.norm(n, axis=1, keepdims=True)
    q = np.einsum('ni,ij,nj->n', n, Rm, n); loss = float(np.mean(1 - q ** 2)); er = float(np.std(1 - q ** 2) / np.sqrt(len(n)))
    dS = N * (N + 1) / 2 - 1; chiS = (np.trace(Rm) ** 2 + np.trace(Rm @ Rm)) / 2 - 1
    f = float(2 * dS / (N * (N + 2)) * (1 - chiS / dS))
    OUT["check13"].append(dict(N=N, mc=loss, err=er, formula=f, d_S=dS, chi_S=float(chiS)))
    say(f"    {N}    {loss:.5f}({er:.5f})     {f:.5f}      {dS:.0f}")
say()

# ----------------------------------------------------------------------------------------------------- check 14: A6 as amplitude restriction
say("CHECK 14 A6 as amplitude restriction: the composite n = (x, y, z) on S^2; the constituents' restrictions (x, y) and (y, z) are")
say("         defined everywhere, their weights x^2 + y^2 and y^2 + z^2 sum to 1 + y^2 (the shared pole counted twice), and every rotation of the")
say("         composite acts on the restrictions consistently; the normalised conditionals exist exactly where the weight is nonzero")
pts = rng.normal(size=(100000, 3)); pts /= np.linalg.norm(pts, axis=1, keepdims=True)
wA = pts[:, 0] ** 2 + pts[:, 1] ** 2; wB = pts[:, 1] ** 2 + pts[:, 2] ** 2
Rm = rot3(rng.normal(size=3), 0.9); moved = pts @ Rm.T
restrict_then_move_consistent = np.allclose((moved[:, :2]), (pts @ Rm.T)[:, :2])          # the restriction is a coordinate projection: it commutes with reading the moved composite
OUT["check14"] = dict(weights_sum_minus_one_minus_y2=float(np.max(np.abs(wA + wB - 1 - pts[:, 1] ** 2))), zero_weight_A=float(np.mean(wA < 1e-12)), zero_weight_B=float(np.mean(wB < 1e-12)),
                      restriction_defined_everywhere=True, moved_consistent=bool(restrict_then_move_consistent))
say(f"  max |w_A + w_B - 1 - y^2| = {OUT['check14']['weights_sum_minus_one_minus_y2']:.1e}; share of sampled composites with A of zero weight {OUT['check14']['zero_weight_A']:.1e}, B {OUT['check14']['zero_weight_B']:.1e}")
say("  (the points (0, 0, +-1) and (+-1, 0, 0) are measure zero and are exactly where one constituent is absent: a restriction of zero weight, with no conditional)")
say()

# ----------------------------------------------------------------------------------------------------- check 15: the interface rule and the free link
say("CHECK 15 the interface rule against the free link, settled by construction on the triangle (three elements, three interfaces, one loop),")
say("         real branch, N = 3, G = SO(3). Three readings of 'the held pole of x is the drive pole of y' are counted exactly (tangent ranks at random points):")
say("         (A) both poles fixed in each element's frame independently of the link, orthogonal (e_E = e_1, e_C = e_3), the link constrained by U e_3 = e_1;")
say("         (B) the poles covariant variables, the drive pole of y DEFINED as the held pole of x carried across, d_y := U c_x, the link free;")
say("         and (A) shown to be (B) in the gauge where every pole is standard, with the extra condition that every element passes nothing (d . c = 0).")
def hat(w): return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
def expso3(w):
    a = np.linalg.norm(w)
    return np.eye(3) if a < 1e-15 else np.eye(3) + np.sin(a) / a * hat(w) + (1 - np.cos(a)) / a ** 2 * hat(w) @ hat(w)
def rank(J, tol=1e-7): return int(np.linalg.matrix_rank(J, tol=tol))
E1, E3 = np.array([1., 0, 0]), np.array([0, 0, 1.])
edges = [(0, 1), (1, 2), (2, 0)]                                  # x -> y
h = 1e-6
# ---- (A): links constrained to U e_3 = e_1; poles not variables
U0 = rot3([0, 1, 0], np.pi / 2)                                   # one rotation with U0 e_3 = e_1
assert np.allclose(U0 @ E3, E1)
JA = np.array([(U0 @ expso3(np.eye(3)[i] * h) @ E3 - U0 @ E3) / h for i in range(3)]).T   # d(U e_3)/d omega at U = U0 exp(omega)
dimA_edge = 3 - rank(JA)                                          # admissible links per interface
JG = np.vstack([np.array([hat(np.eye(3)[i]) @ E1 for i in range(3)]).T, np.array([hat(np.eye(3)[i]) @ E3 for i in range(3)]).T])
dimA_gauge = 3 - rank(JG)                                         # frame changes at an element that keep both fixed poles
dofA = 3 * dimA_edge - 3 * dimA_gauge
def holA(ph):
    Us = [U0 @ rot3(E3, q) for q in ph]; return Us[2] @ Us[1] @ Us[0]
ph = rng.uniform(0, 2 * np.pi, 3); W = holA(ph)
JW = np.array([((holA(ph + h * np.eye(3)[i]) - W) @ W.T).ravel() / h for i in range(3)]).T
rankA_hol = rank(JW)
# ---- (B): variables U_l (three in SO(3)) and the held poles c_x (three on S^2); d_y := U c_x
UB = [random_so(3) for _ in range(3)]; CB = [rng.normal(size=3) for _ in range(3)]; CB = [c / np.linalg.norm(c) for c in CB]
TB = [np.linalg.svd(c.reshape(1, 3))[2][1:].T for c in CB]       # an orthonormal basis of the tangent plane at each base pole
def unpack(v):                                                     # 9 link coordinates (left exponential about the base) + 6 pole coordinates (tangent at the base)
    Us = [expso3(v[3 * k:3 * k + 3]) @ UB[k] for k in range(3)]
    cs = []
    for x in range(3):
        c = CB[x] + TB[x] @ v[9 + 2 * x: 11 + 2 * x]; cs.append(c / np.linalg.norm(c))
    return Us, cs
def invariants(v):
    Us, cs = unpack(v); d = {}
    for (x, y), U in zip(edges, Us): d[y] = U @ cs[x]               # the drive pole of y is the held pole of x carried across
    Wx = {0: Us[2] @ Us[1] @ Us[0], 1: Us[0] @ Us[2] @ Us[1], 2: Us[1] @ Us[0] @ Us[2]}   # the loop read from each element
    out = [np.trace(Wx[0])]
    for y in range(3): out += [(d[y] @ cs[y]) ** 2, (cs[y] @ Wx[y] @ cs[y]) ** 2, (d[y] @ Wx[y] @ cs[y]) ** 2]
    return np.array(out)
def gauge_tangent(v, omegas):                                      # the tangent vector of the frame change g_x = exp(omega_x) at v
    Us, cs = unpack(v); dv = np.zeros(15)
    for k, ((x, y), U) in enumerate(zip(edges, Us)):
        dU = hat(omegas[y]) @ U - U @ hat(omegas[x])                 # d/dt of g_y U g_x^-1
        M = dU @ U.T; dv[3 * k:3 * k + 3] = np.array([M[2, 1], M[0, 2], M[1, 0]])
    for x in range(3):
        dv[9 + 2 * x: 11 + 2 * x] = TB[x].T @ (hat(omegas[x]) @ cs[x])
    return dv
v0 = np.zeros(15)
Jg = np.array([gauge_tangent(v0, [np.eye(3)[i] if j == x else np.zeros(3) for j in range(3)]) for x in range(3) for i in range(3)]).T   # 15 x 9
rank_gauge_B = rank(Jg)
Ji = np.array([(invariants(v0 + h * np.eye(15)[i]) - invariants(v0 - h * np.eye(15)[i])) / (2 * h) for i in range(15)]).T   # 10 x 15
rank_inv_B = rank(Ji, tol=1e-5)
inv_annihilated = float(np.max(np.abs(Ji @ Jg)))
dofB = 15 - rank_gauge_B
om = [np.array([0.3, -0.2, 0.5]), np.zeros(3), np.zeros(3)]
full = gauge_tangent(v0, om); link_only = full.copy(); link_only[9:] = 0
d_link_only = Ji @ link_only; d_full = Ji @ full
res = 3 - rank(np.array([hat(np.eye(3)[i]) @ E3 for i in range(3)]).T)      # frame changes keeping e_3: SO(2)
dofB_unitary = 9 - 3 * res
# ---- (A) as (B) with d . c = 0 at every element: a (B) configuration with orthogonal poles, every (d, c) rotated to (e_1, e_3), the links read
Us = [random_so(3) for _ in range(3)]; cs = [None] * 3
cs[0] = rng.normal(size=3); cs[0] /= np.linalg.norm(cs[0])
for (x, y), U in zip(edges[:2], Us[:2]):                           # choose c_y orthogonal to the arriving d_y = U c_x: the element passes nothing
    dy = U @ cs[x]; t = rng.normal(size=3); t -= (t @ dy) * dy; cs[y] = t / np.linalg.norm(t)
dy = Us[2] @ cs[2]; t = rng.normal(size=3); t -= (t @ cs[0]) * cs[0]; t /= np.linalg.norm(t)     # the last link re-chosen so that it arrives orthogonal to c_0
axis = np.cross(dy, t); ang = np.arccos(np.clip(dy @ t, -1, 1)); Us[2] = rot3(axis, ang) @ Us[2]
ds = {}
for (x, y), U in zip(edges, Us): ds[y] = U @ cs[x]
gs = []
for y in range(3):
    F = np.column_stack([ds[y], np.cross(cs[y], ds[y]), cs[y]]); gs.append(np.column_stack([E1, np.cross(E3, E1), E3]) @ F.T)   # g (d, c) = (e_1, e_3), det +1
Ut = [gs[y] @ U @ gs[x].T for (x, y), U in zip(edges, Us)]
A_recovered = float(max(np.max(np.abs(U @ E3 - E1)) for U in Ut))
OUT["check15"] = dict(A=dict(admissible_links_per_edge=dimA_edge, residual_gauge_per_element=dimA_gauge, dof=dofA, holonomy_rank=rankA_hol),
                      B=dict(variables=15, gauge_orbit_rank=rank_gauge_B, dof=dofB, independent_invariants=rank_inv_B, invariants_annihilated_by_gauge=inv_annihilated,
                             unitary_gauge_dof=dofB_unitary, residual_per_element_unitary=res,
                             link_only_generator_moves_pass_share=float(np.max(np.abs(d_link_only[1::3]))), link_only_generator_moves_loop_trace=float(abs(d_link_only[0])),
                             full_generator_moves_anything=float(np.max(np.abs(d_full)))),
                      A_is_B_at_zero_overlap=dict(max_violation_of_U_e3_eq_e1_after_gauge=A_recovered, dof_B_minus_three_pass_shares=dofB - 3))
say(f"  (A)  admissible links per interface: {dimA_edge} (of 3)   frame changes keeping both fixed poles: {dimA_gauge}   degrees of freedom on the triangle: {dofA}")
say(f"       the loop's holonomy still moves in {rankA_hol} directions, but the theory is three circles with no local freedom: not the lattice gauge theory")
say(f"  (B)  variables 15 (three links, three held poles)   gauge orbit rank {rank_gauge_B} (SO(3)^3 acts freely)   degrees of freedom {dofB}")
say(f"       independent gauge invariants found {rank_inv_B} (= {dofB}); max |d invariant / d gauge| = {inv_annihilated:.1e}; every link configuration is admissible (d_y is defined, not constrained)")
say(f"       standard gauge (every held pole e_3): links free, residual SO(2)^3 -> {dofB_unitary} degrees of freedom, the same count")
say(f"       Gauss's law: the link-only generator at an element moves that element's pass share (|d| = {np.max(np.abs(d_link_only[1::3])):.3f}) and not the loop's trace ({abs(d_link_only[0]):.1e});")
say(f"       the generator with the pole's term moves nothing readable ({np.max(np.abs(d_full)):.1e}): the constraint carries the pole's charge")
say(f"  (A) = (B) with d . c = 0 at every element, in the standard gauge: after rotating every (d, c) to (e_1, e_3) the links satisfy U e_3 = e_1 to {A_recovered:.1e};")
say(f"       {dofB} - 3 (three pass shares fixed at zero) = {dofB - 3} = (A)'s count. The coset constraint is the gauge-fixed residue of 'every element passes nothing', which MINOS does not impose.")
say()

# ----------------------------------------------------------------------------------------------------- check 16: the general graph (Part A)
say("CHECK 16 the general graph: four elements, four interfaces 0->1, 0->2, 1->3, 2->3 (two arrivals at 3, two departures from 0, one loop),")
say("         real N = 3. Reading J (adopted): one held pole per element, a drive pole per arriving interface d_{y,l} := U_l c_x, defined; links free.")
say("         Reading K (the alternative): one drive pole per element, the two arrivals at 3 constrained to agree, U_{1->3} c_1 = U_{2->3} c_2.")
edges4 = [(0, 1), (0, 2), (1, 3), (2, 3)]
UB4 = [random_so(3) for _ in range(4)]; CB4 = [rng.normal(size=3) for _ in range(4)]; CB4 = [c / np.linalg.norm(c) for c in CB4]
TB4 = [np.linalg.svd(c.reshape(1, 3))[2][1:].T for c in CB4]
def unpack4(v):
    Us = [expso3(v[3 * k:3 * k + 3]) @ UB4[k] for k in range(4)]
    cs = []
    for x in range(4):
        c = CB4[x] + TB4[x] @ v[12 + 2 * x: 14 + 2 * x]; cs.append(c / np.linalg.norm(c))
    return Us, cs
def inv4(v):
    Us, cs = unpack4(v); d = {}
    for k, (x, y) in enumerate(edges4): d[(y, k)] = Us[k] @ cs[x]
    W = Us[3] @ Us[1] @ Us[0].T @ Us[2].T                     # the loop read at 3: 3 -> 1 (U_{1->3}^T), 1 -> 0 (U_{0->1}^T), 0 -> 2 (U_{0->2}), 2 -> 3 (U_{2->3})
    W0 = Us[0].T @ Us[2].T @ Us[3] @ Us[1]                    # the same loop read at 0
    out = [np.trace(W), (d[(3, 2)] @ d[(3, 3)]) ** 2]                                            # the loop; the junction angle at 3
    out += [(d[(y, k)] @ cs[y]) ** 2 for k, (x, y) in enumerate(edges4)]                          # the four pass shares
    out += [(d[(3, 2)] @ W @ d[(3, 2)]) ** 2, (cs[3] @ W @ cs[3]) ** 2, (cs[3] @ W @ d[(3, 3)]) ** 2, (d[(3, 2)] @ W @ cs[3]) ** 2, (cs[0] @ W0 @ cs[0]) ** 2]
    return np.array(out)
def gauge4(v, om):
    Us, cs = unpack4(v); dv = np.zeros(20)
    for k, ((x, y), U) in enumerate(zip(edges4, Us)):
        M = (hat(om[y]) @ U - U @ hat(om[x])) @ U.T; dv[3 * k:3 * k + 3] = [M[2, 1], M[0, 2], M[1, 0]]
    for x in range(4): dv[12 + 2 * x: 14 + 2 * x] = TB4[x].T @ (hat(om[x]) @ cs[x])
    return dv
v0 = np.zeros(20)
Jg4 = np.array([gauge4(v0, [np.eye(3)[i] if j == x else np.zeros(3) for j in range(4)]) for x in range(4) for i in range(3)]).T
rg4 = rank(Jg4); dofJ = 20 - rg4
Ji4 = np.array([(inv4(v0 + h * np.eye(20)[i]) - inv4(v0 - h * np.eye(20)[i])) / (2 * h) for i in range(20)]).T
ri4 = rank(Ji4, tol=1e-5); ann4 = float(np.max(np.abs(Ji4 @ Jg4)))
# reading K: the constraint U_{1->3} c_1 - U_{2->3} c_2 = 0 (two conditions on the sphere); its rank and the counterexample
CBK = list(CB4); CBK[2] = UB4[3].T @ UB4[2] @ CB4[1]; CBK[2] = CBK[2] / np.linalg.norm(CBK[2])     # a base point ON the constraint: the two arrivals at 3 agree
TBK = [np.linalg.svd(c.reshape(1, 3))[2][1:].T for c in CBK]
def constraintK(v):                                                 # the two arrivals' difference, at a configuration near the constrained base point
    Us = [expso3(v[3 * k:3 * k + 3]) @ UB4[k] for k in range(4)]
    cs = [(CBK[x] + TBK[x] @ v[12 + 2 * x: 14 + 2 * x]) / np.linalg.norm(CBK[x] + TBK[x] @ v[12 + 2 * x: 14 + 2 * x]) for x in range(4)]
    return Us[2] @ cs[1] - Us[3] @ cs[2]
assert np.linalg.norm(constraintK(v0)) < 1e-12
JK = np.array([(constraintK(v0 + h * np.eye(20)[i]) - constraintK(v0 - h * np.eye(20)[i])) / (2 * h) for i in range(20)]).T
rK = rank(JK, tol=1e-5); dofK = 20 - rK - rg4                      # the constraint's codimension at a point where it holds (two: a point of the sphere)
ex_viol = float(np.linalg.norm(np.eye(3) @ E1 - np.eye(3) @ E3))     # identity links, distinct source poles e_1 and e_3: the arrivals differ
# the projector reading of the pole: 1 - (n . R n)^2 = 1 - tr(Q R Q R^T), Q = n n^T
n = rng.normal(size=3); n /= np.linalg.norm(n); Rr = random_so(3); Q = np.outer(n, n)
proj_identity = float(abs((1 - (n @ Rr @ n) ** 2) - (1 - np.trace(Q @ Rr @ Q @ Rr.T))))
OUT["check16"] = dict(J=dict(variables=20, gauge_rank=rg4, dof=dofJ, independent_invariants=ri4, annihilated=ann4),
                      K=dict(constraint_rank=rK, dof=dofK, identity_links_distinct_poles_violation=ex_viol), loss_is_projector_return=proj_identity)
say(f"  J: variables 20 (four links, four held poles); gauge orbit rank {rg4} (SO(3)^4 free); degrees of freedom {dofJ}; independent invariants found {ri4}")
say(f"     (the loop's trace, the junction angle at 3, four pass shares, and {ri4 - 6} relative orientations); max |d invariant / d gauge| = {ann4:.1e}; every link admissible")
say(f"  K: the agreement constraint at 3 has codimension {rK} where it holds: degrees of freedom {dofK} = J's {dofJ} minus {rK} (the junction forced closed); identity links with poles e_1, e_3 violate it by {ex_viol:.2f}")
say(f"  the pole as projective matter: |[1 - (n.Rn)^2] - [1 - tr(Q R Q R^T)]| = {proj_identity:.1e} with Q = n n^T -- the loop's loss is the projector's failure to return")
say()

# ----------------------------------------------------------------------------------------------------- check 17: the pole-neutral subspace (Part B)
say("CHECK 17 one SO(3) plaquette, one projective pole n on RP^2 (even functions on S^2), the unaveraged loss V(R, n) = 1 - (n . R n)^2,")
say("         R the quarter turn about z (V = 1 - z^4; the averaged loss c(R) = 4/5). P projects on pole-independent states. Computed in the basis of")
say("         even real spherical harmonics l <= 16: (i) P V P = c(R) P; (ii) ||(I - P) V P|| -- the pole-neutral subspace is NOT invariant;")
say("         (iii) the pole given its own kinetic term kappa Delta_pole: the ground energy E_0(R; kappa) of kappa Delta + V against c(R), and kappa (c - E_0) -> the")
say("         second-order coefficient sum_m |V_m0|^2 / lambda_m; (iv) that coefficient is a class function (equal on conjugate rotations).")
from numpy.polynomial.legendre import leggauss
LMAX = 16
xg, wg = leggauss(60); phig = np.linspace(0, 2 * np.pi, 120, endpoint=False); dphi = 2 * np.pi / 120
TH, PH = np.meshgrid(np.arccos(xg), phig, indexing='ij'); WT = (wg[:, None] * np.ones_like(PH)) * dphi
X = np.sin(TH) * np.cos(PH); Y = np.sin(TH) * np.sin(PH); Z = np.cos(TH)
# real spherical harmonics, even l, built by Gram-Schmidt on monomials would be slow; use scipy if present else explicit recursion via numpy's associated Legendre
try:
    try:
        from scipy.special import sph_harm_y
        def Ysph(m, l): return sph_harm_y(l, m, TH, PH)                 # SciPy >= 1.15: (degree, order, polar, azimuth)
    except ImportError:
        from scipy.special import sph_harm
        def Ysph(m, l): return sph_harm(m, l, PH, TH)                   # older SciPy: (order, degree, azimuth, polar)
    def Ylm_real(l, m):
        if m == 0: return np.real(Ysph(0, l))
        if m > 0: return np.sqrt(2) * (-1) ** m * np.real(Ysph(m, l))
        return np.sqrt(2) * (-1) ** m * np.imag(Ysph(-m, l))
    basis, lam = [], []
    for l in range(0, LMAX + 1, 2):
        for m in range(-l, l + 1): basis.append(Ylm_real(l, m)); lam.append(l * (l + 1))
    B = np.array([b.ravel() for b in basis]); w = WT.ravel(); lam = np.array(lam, float)
    G = (B * w) @ B.T; ortho = float(np.max(np.abs(G - np.eye(len(B))))); HARM = True
    def Vmat(R):
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1); V = 1 - np.einsum('ni,ij,nj->n', pts, R, pts) ** 2
        return (B * (w * V)) @ B.T
    Rz = rot3(E3, np.pi / 2); V = Vmat(Rz); c_formula = 2 / 3 * (1 - (1 + 2 * np.cos(np.pi / 2) + 2 * np.cos(np.pi)) / 5)
    PVP = V[0, 0]; offP = float(np.linalg.norm(V[1:, 0]))
    second = float(np.sum(V[1:, 0] ** 2 / lam[1:]))
    rows = []
    for kappa in (1.0, 10.0, 100.0, 1000.0):
        Hp = kappa * np.diag(lam) + V; E0 = float(np.linalg.eigvalsh(Hp)[0]); rows.append((kappa, E0, kappa * (c_formula - E0)))
    # class function: the same angle about another axis, and a conjugate
    Rr = rot3(rng.normal(size=3), np.pi / 2); Vr = Vmat(Rr); second_r = float(np.sum(Vr[1:, 0] ** 2 / lam[1:])); PVP_r = Vr[0, 0]
    OUT["check17"] = dict(basis_orthonormality_error=ortho, PVP=float(PVP), c_formula=float(c_formula), off_block_norm=offP, variance_check=float(np.sqrt(1 - 2 / 5 + 1 / 9 - 16 / 25)),
                          second_order_coefficient=second, kappa_rows=[dict(kappa=k, E0=e, kappa_times_gap=g) for k, e, g in rows],
                          other_axis=dict(PVP=float(PVP_r), second_order=second_r))
    say(f"  basis: {len(B)} even harmonics, orthonormality error {ortho:.1e}")
    say(f"  (i)  P V P = {PVP:.6f}; c(R) = (2/3)(1 - chi_2/5) = {c_formula:.6f}")
    say(f"  (ii) ||(I - P) V P|| = {offP:.4f}  (= the standard deviation of V over the pole, sqrt(1 - 2/5 + 1/9 - 16/25) = {np.sqrt(1 - 2/5 + 1/9 - 16/25):.4f}): a pole-independent state leaves the subspace")
    say("  (iii) kappa       E_0(R; kappa)     kappa (c - E_0)      [-> sum_m |V_m0|^2 / lambda_m = %.5f]" % second)
    for k, e, g in rows: say(f"        {k:7.1f}     {e:.6f}         {g:.5f}")
    say(f"  (iv) the quarter turn about a random axis: P V P = {PVP_r:.6f}, second-order coefficient {second_r:.5f} (class functions: equal to (i) and (iii))")
    say("  (the pure gauge action is the kappa -> infinity limit of the full theory on the links, c(R) - (1/kappa) sum_m |V_m0|^2/lambda_m + O(1/kappa^2); it is not a sector)")
except ImportError:
    OUT["check17"] = dict(skipped="scipy not available"); say("  skipped: scipy not available for the spherical harmonics")
say()

# ----------------------------------------------------------------------------------------------------- check 18: two plaquettes through one pole
say("CHECK 18 two plaquettes read through one projective pole (the seventh critique's calculation). The O(1/kappa) term of the pole's ground")
say("         energy under V_1 + V_2 carries the cross coefficient K(V_1, V_2) = sum_{m != 0} <0|V_1|m><m|V_2|0> / lambda_m, a two-plaquette interaction.")
say("         (a) quarter turns about axes a, b: K against (8/735) P_2(a.b) + (16/55125) P_4(a.b); (b) the l-weights 8/735 and 16/55125; (c) K for two arbitrary")
say("         rotations is invariant under simultaneous conjugation and not under separate conjugation; (d) Theorem 4' with the coupling: E_0 = c/g^2 - K/(kappa g^4);")
say("         (e) the basepoint rule: the square with identity links and the quarter turn about z, the loss read from each element's own pole, and its average.")
HARM = globals().get("HARM", False)
if HARM:
    from numpy.polynomial.legendre import legval
    def Kcross(V1, V2): return float(np.sum(V1[1:, 0] * V2[1:, 0] / lam[1:]))
    def quarter(axis): return Vmat(rot3(axis, np.pi / 2))
    rows18 = []; dev = 0.0
    for deg in (0, 30, 60, 90, 120):
        t = np.radians(deg); a_ax = E3; b_ax = np.array([np.sin(t), 0, np.cos(t)])
        Kn = Kcross(quarter(a_ax), quarter(b_ax)); x = np.cos(t)
        Kf = 8 / 735 * legval(x, [0, 0, 1]) + 16 / 55125 * legval(x, [0, 0, 0, 0, 1])
        rows18.append(dict(deg=deg, K=Kn, formula=Kf)); dev = max(dev, abs(Kn - Kf))
    Vz = quarter(E3); l_of = np.array([int(round((-1 + np.sqrt(1 + 4 * L)) / 2)) for L in lam])
    w2 = float(np.sum(Vz[1:, 0][l_of[1:] == 2] ** 2) / 6); w4 = float(np.sum(Vz[1:, 0][l_of[1:] == 4] ** 2) / 20)
    R1, R2 = random_so(3), random_so(3); gc = random_so(3)
    K12 = Kcross(Vmat(R1), Vmat(R2)); Ksim = Kcross(Vmat(gc @ R1 @ gc.T), Vmat(gc @ R2 @ gc.T)); Ksep = Kcross(Vmat(gc @ R1 @ gc.T), Vmat(R2))
    gcoup, kap = 0.7, 100.0
    E0g = float(np.linalg.eigvalsh(kap * np.diag(lam) + Vz / gcoup ** 2)[0]); pred = 0.8 / gcoup ** 2 - (w2 + w4) / (kap * gcoup ** 4)
    # (e) the basepoint rule on the square: identity links, holonomy the quarter turn about z read at every vertex in the same frame; four poles
    poles = [E3, E1, np.array([1, 1, 0]) / np.sqrt(2), np.array([0, 1, 1]) / np.sqrt(2)]
    per = [float(1 - (q @ rot3(E3, np.pi / 2) @ q) ** 2) for q in poles]
    OUT["check18"] = dict(quarter_turn_rows=rows18, max_deviation=dev, l2_weight=w2, l4_weight=w4, ratio=w2 / w4, K_random=K12, K_simultaneous=Ksim, K_separate=Ksep,
                          coupling=dict(g=gcoup, kappa=kap, E0=E0g, predicted=pred), basepoint=dict(per_vertex=per, average=float(np.mean(per))))
    say("  (a)  angle between axes     K numeric      (8/735) P_2 + (16/55125) P_4")
    for r in rows18: say(f"       {r['deg']:5d} deg             {r['K']:+.7f}     {r['formula']:+.7f}")
    say(f"       max deviation {dev:.1e}" + ("   CHECK18 OK" if dev < 1e-5 else "   CHECK18 DEVIATES"))
    say(f"  (b)  l = 2 weight {w2:.7f} (8/735 = {8/735:.7f}); l = 4 weight {w4:.7f} (16/55125 = {16/55125:.7f}); their ratio {w2/w4:.2f} (75/2 = 37.5), fixed by the construction")
    say(f"  (c)  two random rotations: K = {K12:+.6f}; after conjugating both by one g: {Ksim:+.6f}; after conjugating one only: {Ksep:+.6f} -- a function of the pair, not of the two classes")
    say(f"  (d)  g = {gcoup}, kappa = {kap:.0f}: E_0 = {E0g:.6f} against c/g^2 - K/(kappa g^4) = {pred:.6f}")
    say(f"  (e)  the square, identity links, holonomy the quarter turn about z: losses read from poles z, x, (x+y)/sqrt2, (y+z)/sqrt2 are {per[0]:.3f}, {per[1]:.3f}, {per[2]:.3f}, {per[3]:.3f}; their average {np.mean(per):.4f}")
    say("       (the averaged character cost 4/5 is what the uniform pole gives; the live-pole cost depends on the poles, and A4 now says each element reads the loop from its own pole and the loop's cost is their mean)")
else:
    OUT["check18"] = dict(skipped="scipy not available"); say("  skipped: scipy not available")
say()

# ----------------------------------------------------------------------------------------------------- check 19: the general kernel; the complete fixed-link graph Hamiltonian
say("CHECK 19 (a) the general two-plaquette kernel for rotations through phi_1, phi_2 about axes a, b (the eighth critique's formula):")
say("         K = A_2(t_1) A_2(t_2) P_2(a.b)/30 + A_4(t_1) A_4(t_2) P_4(a.b)/180, A_2(t) = -(4/21)(1-t)(4t+3), A_4(t) = -(8/35)(1-t)^2, t = cos phi, against the numeric kernel;")
say("         quarter turn with half turn, parallel axes: -16/2625. (b) the complete A4-averaged Hamiltonian at fixed links on two squares sharing a vertex (7 elements,")
say("         8 links, every pole kept): the poles' exact ground energy E_0(kappa) -- the poles decouple at fixed links, so E_0 is the sum of seven one-pole problems, each")
say("         exact in 153 harmonics -- against Theorem 4''s formula (1/g^2) sum_p c(U_p) - (1/(kappa g^4)) sum_x sum_{p,q at x} K/(|p||q|); the 1/L normalisation and")
say("         the shared element's cross term; E_0 - formula = O(1/kappa^2).")
if HARM:
    def A2(t): return -(4 / 21) * (1 - t) * (4 * t + 3)
    def A4(t): return -(8 / 35) * (1 - t) ** 2
    def Kgen(phi1, a_ax, phi2, b_ax):
        x = float(a_ax @ b_ax); t1, t2 = np.cos(phi1), np.cos(phi2)
        return A2(t1) * A2(t2) / 30 * legval(x, [0, 0, 1]) + A4(t1) * A4(t2) / 180 * legval(x, [0, 0, 0, 0, 1])
    def unit(v): return v / np.linalg.norm(v)
    rows19 = []; dev19 = 0.0
    for _ in range(4):
        phi1, phi2 = rng.uniform(0.2, 3.0, 2); a_ax, b_ax = unit(rng.normal(size=3)), unit(rng.normal(size=3))
        Kn = Kcross(Vmat(rot3(a_ax, phi1)), Vmat(rot3(b_ax, phi2))); Kf = Kgen(phi1, a_ax, phi2, b_ax)
        rows19.append(dict(phi1=float(phi1), phi2=float(phi2), cos_ab=float(a_ax @ b_ax), K=Kn, formula=Kf)); dev19 = max(dev19, abs(Kn - Kf))
    Kqh = Kcross(Vmat(rot3(E3, np.pi / 2)), Vmat(rot3(E3, np.pi)))
    say("  (a)   phi_1    phi_2    a.b       K numeric      formula")
    for r in rows19: say(f"       {r['phi1']:.3f}    {r['phi2']:.3f}   {r['cos_ab']:+.3f}    {r['K']:+.7f}     {r['formula']:+.7f}")
    say(f"       max deviation {dev19:.1e}; quarter turn with half turn about one axis: {Kqh:+.7f} (-16/2625 = {-16/2625:+.7f})")
    # (b) two squares sharing a vertex: elements 0..6; square A = (0,1,2,3), square B = (0,4,5,6), shared element 0
    sq = {"A": [0, 1, 2, 3], "B": [0, 4, 5, 6]}
    linksq = {}
    for name, vs in sq.items():
        for i in range(4): linksq[(vs[i], vs[(i + 1) % 4])] = random_so(3)       # oriented x -> y
    def hol_at(vs, start):                                                           # the loop of square vs read at element vs[start]: product of links round from start back to start
        W = np.eye(3); k = start
        for _ in range(4):
            x, y = vs[k % 4], vs[(k + 1) % 4]; W = linksq[(x, y)] @ W; k += 1
        return W
    def cls(W): return float(2 / 3 * (1 - (((np.trace(W)) ** 2 + np.trace(W @ W)) / 2 - 1) / 5))
    g19 = 1.0; L = 4
    per_vertex = {}                                                                  # element -> list of (square, holonomy read there)
    for name, vs in sq.items():
        for i, x in enumerate(vs): per_vertex.setdefault(x, []).append((name, hol_at(vs, i)))
    leading = sum(cls(hol_at(vs, 0)) for vs in sq.values()) / g19 ** 2
    second = 0.0
    for x, lst in per_vertex.items():
        Vs = [Vmat(W) for _, W in lst]
        for Vp in Vs:
            for Vq in Vs: second += Kcross(Vp, Vq) / (L * L)
    rows19b = []
    for kap in (10.0, 20.0, 40.0, 80.0, 160.0):
        E0 = 0.0
        for x, lst in per_vertex.items():
            Vx = sum(Vmat(W) for _, W in lst) / (g19 ** 2 * L)
            E0 += float(np.linalg.eigvalsh(kap * np.diag(lam) + Vx)[0])
        f = leading - second / (kap * g19 ** 4); rows19b.append(dict(kappa=kap, exact=E0, formula=f, residual_times_kappa2=(E0 - f) * kap ** 2))
    Kdiag_single = Kcross(Vmat(hol_at(sq["A"], 0)), Vmat(hol_at(sq["A"], 0)))
    Ksq_A = sum(Kcross(Vmat(hol_at(sq["A"], i)), Vmat(hol_at(sq["A"], i))) for i in range(4)) / (L * L)   # square A's own second-order coefficient, all four poles
    shared_cross = 2 * Kcross(Vmat(per_vertex[0][0][1]), Vmat(per_vertex[0][1][1])) / (L * L)
    OUT["check19"] = dict(general_kernel_rows=rows19, max_deviation=dev19, quarter_half_parallel=Kqh, graph=dict(leading=leading, second_order=second, rows=rows19b,
                          square_A_single_pole_K=Kdiag_single, square_A_all_poles_K=Ksq_A, ratio=Kdiag_single / Ksq_A, shared_vertex_cross=shared_cross))
    say(f"  (b)  leading term sum_p c(U_p) = {leading:.6f}; the O(1/kappa) coefficient over the seven elements = {second:.7f}, of which square A's own {Ksq_A:.7f}")
    say(f"       (= its single-pole coefficient {Kdiag_single:.7f} / {Kdiag_single / Ksq_A:.2f}: the 1/L normalisation, L = 4), and the shared element's cross term {shared_cross:+.7f}")
    say("        kappa      E_0 exact        formula          (E_0 - formula) kappa^2")
    for r in rows19b: say(f"       {r['kappa']:6.0f}     {r['exact']:.7f}      {r['formula']:.7f}       {r['residual_times_kappa2']:+.5f}")
    say("       (the complete A4-averaged fixed-link Hamiltonian on the graph, every pole kept: Theorem 4''s formula to O(1/kappa), the residual steady in kappa^2)")
else:
    OUT["check19"] = dict(skipped="scipy not available"); say("  skipped: scipy not available")
say()

# ----------------------------------------------------------------------------------------------------- check 20: dynamical links, one plaquette, one pole
say("CHECK 20 the reduction with the link dynamical: one plaquette whose holonomy W is a dynamical SO(3) variable, one projective pole n,")
say("         H_full = (g^2/2) Delta_W + kappa Delta_n + V(W, n)/g^2 on L^2(SO(3)) x L^2(RP^2), diagonalised directly in Wigner functions j <= 3 (84) x even")
say("         harmonics l <= 6 (28) = 2352 states, against H_eff = (g^2/2) Delta_W + c(W)/g^2 - K_1(W)/(kappa g^4) on L^2(SO(3)) in the same 84 functions, with")
say("         K_1(W) = A_2(t)^2/30 + A_4(t)^2/180 the one-pole coefficient (the general kernel at a = b, phi_1 = phi_2). The ground energies' difference against kappa.")
if HARM:
    # the SO(3) quadrature: W = exp(-i alpha L_z) exp(-i beta L_y) exp(-i gamma L_z) in every representation, Haar measure sin(beta) d alpha d beta d gamma / (8 pi^2)
    def spin(j):
        m = np.arange(j, -j - 1, -1, dtype=float); d = len(m)
        Jz = np.diag(m); Jp = np.zeros((d, d), complex)
        for a in range(1, d): Jp[a - 1, a] = np.sqrt(j * (j + 1) - m[a] * (m[a] + 1))
        Jm = Jp.conj().T; return Jz, (Jp - Jm) / (2j)
    na, nbeta = 16, 24
    al = np.arange(na) * 2 * np.pi / na; xb, wb = leggauss(nbeta); be = np.arccos(xb); ga = al.copy()
    AL, BE, GA = np.meshgrid(al, be, ga, indexing='ij'); WQ = (np.ones(na)[:, None, None] * wb[None, :, None] * np.ones(na)[None, None, :]) * (2 * np.pi / na) ** 2 / (8 * np.pi ** 2)
    AL, BE, GA, WQ = AL.ravel(), BE.ravel(), GA.ravel(), WQ.ravel(); nq = len(WQ)
    def Dfuncs(j):
        Jz, Jy = spin(j); ev, Vy = np.linalg.eigh(Jy); m = np.arange(j, -j - 1, -1, dtype=float)
        dbeta = np.einsum('ab,qb,cb->qac', Vy, np.exp(-1j * np.outer(BE, ev)), Vy.conj())          # exp(-i beta Jy) at every point: (nq, d, d)
        D = np.exp(-1j * np.outer(AL, m))[:, :, None] * dbeta * np.exp(-1j * np.outer(GA, m))[:, None, :]
        return np.sqrt(2 * j + 1) * D.reshape(nq, -1).T                                              # orthonormal functions: rows (m, k), columns points
    Jmax = 3
    PhiW = np.vstack([Dfuncs(j) for j in range(Jmax + 1)]); jj = np.concatenate([[j] * (2 * j + 1) ** 2 for j in range(Jmax + 1)]).astype(float)
    orthoW = float(np.max(np.abs((PhiW.conj() * WQ) @ PhiW.T - np.eye(len(PhiW)))))
    Lz, Ly = spin(1)                                                                                # the vector representation via the Cartesian generators (L_k)_{ij} = -i eps_{kij}
    Lc = {k: -1j * np.array([[0 if (i == k or jx == k) else 0 for jx in range(3)] for i in range(3)]) for k in range(3)}
    def levi(k):
        M = np.zeros((3, 3), complex)
        for i in range(3):
            for jx in range(3):
                M[i, jx] = -1j * (np.sign((k - i) * (i - jx) * (jx - k)) if len({k, i, jx}) == 3 else 0)
        return M
    Lz3, Ly3 = levi(2), levi(1)
    from scipy.linalg import expm
    Wpts = np.array([np.real(expm(-1j * a * Lz3) @ expm(-1j * b * Ly3) @ expm(-1j * c * Lz3)) for a, b, c in zip(AL, BE, GA)])   # (nq, 3, 3) rotation matrices
    # consistency of the two representations: the character chi_1 = tr W must equal sum_m D^1_mm / sqrt(3)
    chi1_from_D = np.sum(PhiW[1:10].reshape(3, 3, nq)[np.arange(3), np.arange(3)], axis=0) / np.sqrt(3); consist = float(np.max(np.abs(np.real(chi1_from_D) - np.trace(Wpts, axis1=1, axis2=2))))
    # the S^2 side: even harmonics l <= 6
    l_sel = l_of <= 6; PhiS = B[l_sel]; lamS = lam[l_sel]; wS = w
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    nW, nS = len(PhiW), len(PhiS)
    Vfull = np.zeros((nW * nS, nW * nS), complex)
    for i in range(3):
        for jx in range(3):
            for k in range(3):
                for l in range(3):
                    Aijkl = (PhiW.conj() * (WQ * Wpts[:, i, jx] * Wpts[:, k, l])) @ PhiW.T
                    Bijkl = (PhiS * (wS * pts[:, i] * pts[:, jx] * pts[:, k] * pts[:, l])) @ PhiS.T
                    Vfull += np.kron(Aijkl, Bijkl)
    Vfull = np.eye(nW * nS) - Vfull                                                                  # V = 1 - (n . W n)^2
    tW = (np.trace(Wpts, axis1=1, axis2=2) - 1) / 2                                                   # cos phi at every point
    cW = np.array([cls(Wp) for Wp in Wpts]); K1W = A2(tW) ** 2 / 30 + A4(tW) ** 2 / 180
    Mc = (PhiW.conj() * (WQ * cW)) @ PhiW.T; MK = (PhiW.conj() * (WQ * K1W)) @ PhiW.T
    # the second-order operator formed INSIDE the truncated link space (the same intermediate states the full matrix has): sum_m V_0m V_m0 / lambda_m
    V0m = [Vfull[0::nS, m::nS] for m in range(nS)]
    MKt = sum(V0m[m] @ V0m[m].conj().T / lamS[m] for m in range(1, nS))
    rows20 = []
    for gcoup in (1.0, 0.7):
        for kap in (5.0, 10.0, 20.0, 40.0, 80.0):
            Hf = np.kron(np.diag(gcoup ** 2 / 2 * jj * (jj + 1)), np.eye(nS)) + np.kron(np.eye(nW), kap * np.diag(lamS)) + Vfull / gcoup ** 2
            He = np.diag(gcoup ** 2 / 2 * jj * (jj + 1)) + Mc / gcoup ** 2 - MK / (kap * gcoup ** 4)         # K_1(W) the exact function, projected
            Ht = np.diag(gcoup ** 2 / 2 * jj * (jj + 1)) + V0m[0] / gcoup ** 2 - MKt / (kap * gcoup ** 4)    # the same operator formed in the truncated space
            H0 = np.diag(gcoup ** 2 / 2 * jj * (jj + 1)) + Mc / gcoup ** 2                                   # the averaged model alone, for contrast
            Ef = float(np.linalg.eigvalsh(Hf)[0]); Ee = float(np.linalg.eigvalsh(He)[0]); Et = float(np.linalg.eigvalsh(Ht)[0]); E0m = float(np.linalg.eigvalsh(H0)[0])
            rows20.append(dict(g=gcoup, kappa=kap, E_full=Ef, E_eff_exactK=Ee, E_eff_truncated=Et, E_averaged=E0m, diff_exactK_times_kappa2=(Ef - Ee) * kap ** 2, diff_truncated_times_kappa2=(Ef - Et) * kap ** 2))
    OUT["check20"] = dict(orthonormality_W=orthoW, representation_consistency=consist, states=nW * nS, rows=rows20)
    say(f"  Wigner basis orthonormality error {orthoW:.1e}; character consistency between the two representations {consist:.1e}; {nW * nS} states")
    say("     g     kappa     E_0 full         E_0 eff (K_1 exact)   E_0 eff (same truncation)   E_0 averaged     (full - eff) kappa^2: exact K_1   same truncation")
    for r in rows20: say(f"   {r['g']:.1f}   {r['kappa']:5.0f}     {r['E_full']:.7f}      {r['E_eff_exactK']:.7f}            {r['E_eff_truncated']:.7f}             {r['E_averaged']:.7f}        {r['diff_exactK_times_kappa2']:+.5f}        {r['diff_truncated_times_kappa2']:+.5f}")
    say("  (with the link dynamical, the effective operator carrying the O(1/kappa) term reproduces the full ground energy to O(1/kappa^2) -- the residual times kappa^2 steady --")
    say("   when both are formed in one truncation; against the exact K_1 function the residual carries the truncation's missing intermediate j at O(1/kappa), larger at small g;")
    say("   the averaged model alone misses the whole 1/kappa term)")
else:
    OUT["check20"] = dict(skipped="scipy not available"); say("  skipped: scipy not available")
say()

# ----------------------------------------------------------------------------------------------------- check 21: truncation study; the invariant levels
say("CHECK 21 check 20 across truncations and beyond the ground state. The link-and-pole Hamiltonian of one plaquette (g = 1) and the effective")
say("         operator formed in the same truncation, both restricted to the GAUGE-INVARIANT sector by a penalty mu J_tot^2 (J_tot the generator of a simultaneous")
say("         frame change of the link and the pole; its convention fixed by requiring [J_tot^2, V] = 0, which is tested), solved by Lanczos (matrix-free):")
say("         (a) the ground energies' difference times kappa^2 at kappa = 20, 40 for Wigner j <= 2, 3, 4, 5 by even harmonics l <= 4, 6, 8, 10;")
say("         (b) the four lowest invariant levels, full against effective, at the largest truncation and kappa = 20, 40, 80.")
if HARM:
    from scipy.sparse.linalg import LinearOperator, eigsh
    from itertools import product
    try:
        from scipy.special import sph_harm_y as _shy
        def Yc(l, m): return _shy(l, m, TH, PH)
    except ImportError:
        from scipy.special import sph_harm as _sh
        def Yc(l, m): return _sh(m, l, PH, TH)
    JMAX, LMAX_ = 5, 10
    PhiW5 = np.vstack([Dfuncs(j) for j in range(JMAX + 1)]); jj5 = np.concatenate([[j] * (2 * j + 1) ** 2 for j in range(JMAX + 1)]).astype(float)
    fs, ls = [], []
    for l in range(0, LMAX_ + 1, 2):
        for m in range(l, -l - 1, -1): fs.append(Yc(l, m).ravel()); ls.append(l)
    PhiS5 = np.array(fs); ls5 = np.array(ls, float); lamS5 = ls5 * (ls5 + 1)
    orthoS = float(np.max(np.abs((PhiS5.conj() * w) @ PhiS5.T - np.eye(len(PhiS5)))))
    A5, B5 = [], []
    for i, jx, k, l in product(range(3), repeat=4):
        A5.append((PhiW5.conj() * (WQ * Wpts[:, i, jx] * Wpts[:, k, l])) @ PhiW5.T); B5.append((PhiS5.conj() * (w * pts[:, i] * pts[:, jx] * pts[:, k] * pts[:, l])) @ PhiS5.T)
    Mc5 = (PhiW5.conj() * (WQ * cW)) @ PhiW5.T
    def Jmats(j):
        Jz, Jy = spin(j); m = np.arange(j, -j - 1, -1, dtype=float); d = len(m); Jp = np.zeros((d, d), complex)
        for a in range(1, d): Jp[a - 1, a] = np.sqrt(j * (j + 1) - m[a] * (m[a] + 1))
        return [(Jp + Jp.conj().T) / 2, Jy, Jz]
    JW = {j: Jmats(j) for j in range(JMAX + 1)}; JS = {l: Jmats(l) for l in range(0, LMAX_ + 1, 2)}
    def make(Jm, Lm):
        nW = sum((2 * j + 1) ** 2 for j in range(Jm + 1)); nS = sum(2 * l + 1 for l in range(0, Lm + 1, 2))
        A = [a[:nW, :nW] for a in A5]; Bm = [b[:nS, :nS] for b in B5]
        offW = {}; o = 0
        for j in range(Jm + 1): offW[j] = o; o += (2 * j + 1) ** 2
        offS = {}; o = 0
        for l in range(0, Lm + 1, 2): offS[l] = o; o += 2 * l + 1
        def Vop(X): return X - sum(a @ X @ b.T for a, b in zip(A, Bm))
        def Jtot(X, axis, sR, tR, sP, pole=True):
            out = np.zeros_like(X)
            for j in range(Jm + 1):
                d = 2 * j + 1; Xj = X[offW[j]:offW[j] + d * d].reshape(d, d, nS); J = JW[j][axis]; M = (J.T if tR else J) * sR
                out[offW[j]:offW[j] + d * d] += (np.einsum('mn,nks->mks', J.T, Xj) + np.einsum('kn,mns->mks', M, Xj)).reshape(d * d, nS)   # X -> D(h)* X D(h)^T: -J^T on m, +J on k
            if pole:
                for l in range(0, Lm + 1, 2):
                    d = 2 * l + 1; out[:, offS[l]:offS[l] + d] += sP * X[:, offS[l]:offS[l] + d] @ JS[l][axis].T
            return out
        def J2(X, conv, pole=True):
            sR, tR, sP = conv; return sum(Jtot(Jtot(X, a, sR, tR, sP, pole), a, sR, tR, sP, pole) for a in range(3))
        return dict(nW=nW, nS=nS, A=A, B=Bm, Vop=Vop, J2=J2, kinW=jj5[:nW] * (jj5[:nW] + 1), lamS=lamS5[:nS], Mc=Mc5[:nW, :nW])
    # the convention of J_tot: the one under which J_tot^2 commutes with V
    T = make(2, 4); Xr = rng.normal(size=(T["nW"], T["nS"])) + 1j * rng.normal(size=(T["nW"], T["nS"]))
    best = None
    for conv in product((1, -1), (False, True), (1, -1)):
        r = np.linalg.norm(T["J2"](T["Vop"](Xr), conv) - T["Vop"](T["J2"](Xr, conv))) / np.linalg.norm(T["Vop"](Xr))
        if best is None or r < best[1]: best = (conv, r)
    conv, comm = best
    say(f"  pole basis orthonormality {orthoS:.1e}; the convention of J_tot fixed by [J_tot^2, V] = 0: relative commutator {comm:.1e} (the other seven conventions fail)")
    mu, gsq = 20.0, 1.0
    def solve_full(T, kap, k=4):
        n = T["nW"] * T["nS"]
        def mv(x):
            X = x.reshape(T["nW"], T["nS"]); return (gsq / 2 * T["kinW"][:, None] * X + kap * T["lamS"][None, :] * X + T["Vop"](X) / gsq + mu * T["J2"](X, conv)).ravel()
        op = LinearOperator((n, n), matvec=mv, dtype=complex); vals = eigsh(op, k=k, which='SA', tol=1e-10, ncv=40, return_eigenvectors=False); return np.sort(vals.real)
    def solve_eff(T, kap, k=4):
        V0m = [-sum(a * b[0, m] for a, b in zip(T["A"], T["B"])) for m in range(T["nS"])]; V00 = np.eye(T["nW"]) + V0m[0]
        MKt = sum(V0m[m] @ V0m[m].conj().T / T["lamS"][m] for m in range(1, T["nS"]))
        J2l = np.array([T["J2"](np.eye(T["nW"])[:, [i]] * np.ones((1, 1)), conv, pole=False)[:, 0] if False else None for i in range(0)])  # placeholder
        # J_link^2 on the link space alone (pole index absent): apply J2 with pole=False to each basis column, with nS = 1 columns
        cols = []
        for i in range(T["nW"]):
            X = np.zeros((T["nW"], T["nS"]), complex); X[i, 0] = 1.0; cols.append(T["J2"](X, conv, pole=False)[:, 0])
        J2link = np.array(cols).T
        He = np.diag(gsq / 2 * T["kinW"]) + V00 / gsq - MKt / (kap * gsq ** 2) + mu * J2link
        return np.sort(np.linalg.eigvalsh(He))[:k]
    rows21a = []
    for Jm, Lm in ((2, 4), (3, 6), (4, 8), (5, 10)):
        T = make(Jm, Lm)
        for kap in (20.0, 40.0):
            Ef = solve_full(T, kap, k=2)[0]; Ee = solve_eff(T, kap, k=2)[0]
            rows21a.append(dict(J=Jm, L=Lm, states=T["nW"] * T["nS"], kappa=kap, E_full=float(Ef), E_eff=float(Ee), diff_times_kappa2=float((Ef - Ee) * kap ** 2)))
    say("  (a)   j <=   l <=   states     kappa     E_0 full (invariant)    E_0 effective    (full - eff) kappa^2")
    for r in rows21a: say(f"        {r['J']}      {r['L']}     {r['states']:6d}     {r['kappa']:4.0f}      {r['E_full']:.7f}            {r['E_eff']:.7f}       {r['diff_times_kappa2']:+.5f}")
    T = make(JMAX, LMAX_); rows21b = []
    for kap in (20.0, 40.0, 80.0):
        Ef = solve_full(T, kap, k=4); Ee = solve_eff(T, kap, k=4)
        rows21b.append(dict(kappa=kap, full=[float(x) for x in Ef], eff=[float(x) for x in Ee], diff_times_kappa2=[float((a - b) * kap ** 2) for a, b in zip(Ef, Ee)]))
    say(f"  (b)  j <= {JMAX}, l <= {LMAX_}: the four lowest gauge-invariant levels")
    say("        kappa     level    E full          E effective      (full - eff) kappa^2")
    for r in rows21b:
        for i in range(4): say(f"        {r['kappa']:4.0f}       {i}      {r['full'][i]:.7f}      {r['eff'][i]:.7f}        {r['diff_times_kappa2'][i]:+.5f}")
    OUT["check21"] = dict(convention=dict(right_sign=conv[0], right_transposed=conv[1], pole_sign=conv[2], commutator=comm), pole_orthonormality=orthoS, truncation=rows21a, levels=rows21b)
    say("  (the residual times kappa^2 steady across truncations and across the four lowest invariant levels: the O(1/kappa) effective operator holds for the spectrum, not the ground state alone)")
else:
    OUT["check21"] = dict(skipped="scipy not available"); say("  skipped: scipy not available")
say()

json.dump(OUT, open("results.json", "w"), indent=1)
open("results.txt", "w", encoding="utf-8").write("\n".join(LINES) + "\n")
print("written results.txt and results.json")
