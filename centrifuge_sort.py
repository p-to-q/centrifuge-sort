"""
The most atomic way to simulate and audit Centrifuge Sort in pure, dependency-free Python.
This file is the complete physical algorithm.
Everything else is instrumentation.

@p-to-q / @Jah-yee / ptoq.io
"""

import math
import random
from dataclasses import dataclass


KEYS = [7, 2, 9, 4, 4, 1, 6, 8, 3, 5]

RANDOM_SEED = 1337

def make_keys(n=10, lo=1, hi=9, seed=RANDOM_SEED):
    # Fixed keys make the first run stable.
    # This helper changes the input without adding a CLI.
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n)]


# Let there be scale.
#
# These SI-ish values define a small virtual centrifuge.
# Radius gives the lane, rpm gives the field, viscosity gives resistance.
# The density gradient gives the physical stop, and readout resolution says what the observer can distinguish.

R0 = 0.020                  # m, initial radius
R_WALL = 0.120              # m, outer wall / detector edge

RPM = 3000.0                # rev/min, readable input unit
OMEGA = 2.0 * math.pi * RPM / 60.0  # rad/s, unit used by the equation

ETA = 0.001                 # Pa*s, water-like dynamic viscosity
RHO_FLUID_AT_R0 = 1000.0    # kg/m^3, fluid density near the inner radius

# One model carries both the theory case and the first physical anchor.
#
# If RHO_GRADIENT == 0:
#     the medium is uniform and particles do not self-stop.
#     radius at time T is the readout.
#
# If RHO_GRADIENT > 0:
#     the medium gets denser outward.
#     particles approach rho_particle == rho_fluid(r).
#     equilibrium radius becomes the readout.

RHO_GRADIENT = 2000.0       # kg/m^4, linear density gradient

PARTICLE_DIAMETER = 20e-6   # m, conservative spherical particle size
PARTICLE_RHO_LOW = 1030.0   # kg/m^3, density assigned to the smallest key
PARTICLE_RHO_HIGH = 1180.0  # kg/m^3, density assigned to the largest key

DT = 0.005                  # s, simulation step
STEPS = 6000                # 30 seconds of simulated spin
READOUT_RESOLUTION = 1e-5   # m, closer than this is unresolved by this observer

EPS = 1e-12


# Let there be particles.
#
# A particle carries both id and key.
# The key is the sortable value; the id is the link back to the original record.
# This matters because sorting records is stronger than sorting a value distribution.

@dataclass
class Particle:
    id: int
    key: float
    rho: float
    d: float
    r: float = R0
    v: float = 0.0
    crossed_at: float | None = None


def check_keys(keys):
    if not keys:
        raise ValueError("expected at least one key")

    for x in keys:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise TypeError("keys must be finite int or float values")


# Let there be a medium.
#
# The sorting law is dr/dt = [d^2 * (rho_particle - rho_fluid(r)) / (18 eta)] * omega^2 * r.
# Equal particle and fluid density means neutral buoyancy.
# A positive gradient turns neutral buoyancy into a stable radius.
# This is an overdamped terminal-velocity model, not a full inertial fluid simulation.

def rho_fluid(r):
    return RHO_FLUID_AT_R0 + RHO_GRADIENT * (r - R0)


def sedimentation_term(p, r):
    # Stokes-like term: diameter squared times density contrast divided by viscosity.
    return p.d * p.d * (p.rho - rho_fluid(r)) / (18.0 * ETA)


def radial_velocity(p, r):
    # The centrifugal field contributes omega^2 * r.
    return sedimentation_term(p, r) * OMEGA * OMEGA * r


def equilibrium_radius(p):
    # Linear gradient gives rho_particle == rho_fluid(r); uniform medium has no such stop.
    if RHO_GRADIENT == 0:
        return None
    return R0 + (p.rho - RHO_FLUID_AT_R0) / RHO_GRADIENT


def exact_radius_uniform(p, t):
    # Uniform-medium check: when rho_fluid is constant, r(t) has a closed form.
    s = sedimentation_term(p, R0)
    return R0 * math.exp(s * OMEGA * OMEGA * t)


def reynolds_number(p):
    # Re checks whether the Stokes-like assumption is still plausible for the simulated motion.
    return rho_fluid(p.r) * abs(p.v) * p.d / ETA


def euler_stiffness_scale():
    # This small scale check warns when the chosen dt is too large for the restoring effect of the gradient.
    if RHO_GRADIENT <= 0:
        return 0.0
    return PARTICLE_DIAMETER**2 * RHO_GRADIENT * OMEGA**2 * R_WALL / (18.0 * ETA)


# Let there be keys.
#
# Keys are normalized into [0, 1].
# The normalized value is encoded as particle density.
# Larger key means denser particle, and in this gradient that means larger equilibrium radius.
#
# If two keys are equal:
#     they map to the same density.
#     they are physically indistinguishable in this model.
#     id only gives a stable readout convention.

def encode(keys):
    keys = list(keys)
    check_keys(keys)

    lo, hi = min(keys), max(keys)
    span = hi - lo
    particles = []

    for i, key in enumerate(keys):
        z = 0.5 if span == 0 else (key - lo) / span
        rho = PARTICLE_RHO_LOW + z * (PARTICLE_RHO_HIGH - PARTICLE_RHO_LOW)
        particles.append(Particle(i, key, rho, PARTICLE_DIAMETER))

    return particles


def model_warnings(particles):
    warnings = []

    if RHO_GRADIENT > 0:
        reqs = [equilibrium_radius(p) for p in particles]

        if min(reqs) < R0 - EPS:
            warnings.append("some equilibrium radii are inside R0")

        if max(reqs) > R_WALL + EPS:
            warnings.append("some equilibrium radii exceed R_WALL; top keys may saturate")

        if max(reqs) > R_WALL - READOUT_RESOLUTION:
            warnings.append("largest equilibrium radius is close to the wall")

    else:
        t = STEPS * DT
        predicted = [exact_radius_uniform(p, t) for p in particles]

        if max(predicted) > R_WALL + EPS:
            warnings.append("uniform-medium prediction reaches the wall; top keys may saturate")

    if euler_stiffness_scale() * DT > 0.1:
        warnings.append("Euler step may be coarse for this gradient and rpm")

    return warnings


# Let there be motion.
#
# We integrate with a small Euler step.
# scipy would give better solvers, but it would hide the one equation this file is about.
# Here the update rule stays explicit: radius changes through the particle, the medium, and the centrifugal field.

def detector_threshold():
    return R0 + 0.75 * (R_WALL - R0)


def step(particles, t, dt):
    threshold = detector_threshold()

    for p in particles:
        old_r = p.r
        v = radial_velocity(p, old_r)
        new_r = old_r + v * dt

        if p.crossed_at is None and old_r < threshold <= new_r and new_r != old_r:
            alpha = (threshold - old_r) / (new_r - old_r)
            p.crossed_at = t + alpha * dt

        r = new_r

        if r < R0:
            r = R0
        if r > R_WALL:
            r = R_WALL

        p.v = v
        p.r = r


def run(particles, steps=STEPS, dt=DT):
    t = 0.0
    for _ in range(steps):
        step(particles, t, dt)
        t += dt
    return particles


# Let there be readout.
#
# The centrifuge returns positions rather than a Python list.
# observe_radius() is the ideal observer that reads those positions.
# This step is explicit because readout is part of physical sorting, not background work.
#
# If two radii fall into the same readout bin:
#     the observer cannot separate them by radius.
#     the code keeps id order.
#     the model reports the pair as unresolved.

def radius_bin(p):
    return round((p.r - R0) / READOUT_RESOLUTION)


def observe_radius(particles):
    return sorted(particles, key=lambda p: (radius_bin(p), p.id))


def threshold_detector_events(particles):
    # Earlier events usually correspond to faster outward motion, not ascending sort order.
    return sorted(
        (p for p in particles if p.crossed_at is not None),
        key=lambda p: (p.crossed_at, p.id),
    )


def unresolved_pairs(order):
    pairs = []

    for a, b in zip(order, order[1:]):
        if radius_bin(a) == radius_bin(b) and a.key != b.key:
            pairs.append((a.id, b.id))

    return pairs


# Let there be validation.
#
# sorted() is only the audit oracle.
# If it disagrees with radius readout, the setup failed through encoding, runtime, saturation, or resolution.
# This is a correctness check for the virtual centrifuge, not a benchmark against RAM sorting.

def inversion_count(xs):
    inv = 0

    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] > xs[j]:
                inv += 1

    return inv


def runtime_warnings(order):
    warnings = []

    unresolved = unresolved_pairs(order)
    if unresolved:
        warnings.append("some different keys share a readout bin")

    max_re = max(reynolds_number(p) for p in order)
    if max_re > 1.0:
        warnings.append("Reynolds number exceeds 1; Stokes assumption is weak")

    return warnings


def audit(keys, order, warnings):
    got = [p.key for p in order]
    want = sorted(keys)

    warnings = warnings + runtime_warnings(order)

    return {
        "got": got,
        "want": want,
        "ok": got == want,
        "inversions": inversion_count(got),
        "unresolved": unresolved_pairs(order),
        "max_reynolds": max(reynolds_number(p) for p in order),
        "warnings": warnings,
    }


def cost_ledger(keys):
    keys = list(keys)

    # The ledger is here because physical sorting does not remove cost.
    # It moves cost into encoding, fabrication, motion, precision, space, and readout.
    return {
        "records": len(keys),
        "encoding": "key -> particle density",
        "fabrication": "free in this script; material work in hardware",
        "simulation": f"{len(keys) * STEPS} Python particle updates",
        "physical_evolution": "parallel field if built as hardware",
        "precision": f"{READOUT_RESOLUTION} m radius bins",
        "space": f"{R0} m to {R_WALL} m radial lane",
        "readout": "ideal observer here; sensor or camera in hardware",
    }


def centrifuge_sort(keys=KEYS):
    keys = list(keys)
    particles = encode(keys)
    warnings = model_warnings(particles)

    run(particles)

    order = observe_radius(particles)
    report = audit(keys, order, warnings)

    return order, report


# Let there be presentation.
#
# The C++ version used OpenGL to make the motion visible.
# We keep the visual idea, but replace the runtime dependency with a small ASCII lane.
# The lane is not the algorithm; it is just enough presentation to show that radius became order.

def radial_lane(p, width=48):
    span = R_WALL - R0
    x = round((p.r - R0) / span * (width - 1))
    x = max(0, min(width - 1, x))
    return "|" + "." * x + "*" + "." * (width - 1 - x) + "|"


def print_lanes(order):
    print()
    print("lanes")
    for p in order:
        print(f"id {p.id:2d}  key {p.key:4g}  {radial_lane(p)}")


def print_table(order):
    print()
    print(" id  key      rho_p    rho_f(r)    r_final    v_model        Re   crossed      r_eq")

    for p in order:
        req = equilibrium_radius(p)
        req = "-" if req is None else f"{req:.6f}"

        crossed = "-" if p.crossed_at is None else f"{p.crossed_at:.3f}"

        print(
            f"{p.id:3d} {p.key:4g} "
            f"{p.rho:10.3f} "
            f"{rho_fluid(p.r):10.3f} "
            f"{p.r:10.6f} "
            f"{p.v:10.3g} "
            f"{reynolds_number(p):9.3g} "
            f"{crossed:>9} "
            f"{req:>9}"
        )


def demo(keys=KEYS):
    keys = list(keys)
    order, report = centrifuge_sort(keys)

    print("Centrifuge Sort")
    print("keys -> particles -> spin -> radius -> order")
    print()
    print(f"input:    {keys}")
    print(f"output:   {report['got']}")
    print(f"expected: {report['want']}")
    print(f"ok:       {report['ok']}")
    print(f"errors:   {report['inversions']} inversions")

    for warning in report["warnings"]:
        print(f"warning:  {warning}")

    events = threshold_detector_events(order)
    detector = "none" if not events else " ".join(f"{p.key:g}@{p.crossed_at:.3f}s" for p in events)
    print(f"threshold detector: {detector}")

    print_lanes(order)
    print_table(order)

    ledger = cost_ledger(keys)

    print()
    print("ledger")
    print(f"records:          {ledger['records']}")
    print(f"encoding:         {ledger['encoding']}")
    print(f"fabrication:      {ledger['fabrication']}")
    print(f"simulation:       {ledger['simulation']}")
    print(f"physical step:    {ledger['physical_evolution']}")
    print(f"precision:        {ledger['precision']}")
    print(f"space:            {ledger['space']}")
    print(f"readout:          {ledger['readout']}")

    print()
    print("model")
    print(f"rpm:      {RPM}")
    print(f"omega:    {OMEGA:.3f} rad/s")
    print(f"eta:      {ETA} Pa*s")
    print(f"gradient: {RHO_GRADIENT} kg/m^4")
    print(f"time:     {STEPS * DT:.3f} s")
    print(f"readout:  {READOUT_RESOLUTION} m")
    print(f"max Re:   {report['max_reynolds']:.3g}")


if __name__ == "__main__":
    demo()


# Accepted
# - physical key encoding
# - viscous medium
# - centrifugal field
# - finite readout
# - particle id as record linkage
# - exact uniform-medium solution as theory anchor
# - density gradient as the first physical condition

# Refused
# - findMax / swap / selection-sort-in-disguise
# - "mass alone sorts things"
# - mass / 15 drift
# - hidden O(1) readout
# - claiming that Python simulation is a hardware speedup
# - OpenGL, GLFW, thread loops, and visual runtime as algorithmic dependencies
# - scipy ODE solver in the first file; the first implementation should expose the update before optimizing it

# Limitations
# - spherical particles
# - Newtonian fluid
# - low Reynolds number
# - one ideal radial lane per particle
# - no collisions
# - no aggregation
# - no turbulence
# - no Brownian noise
# - no wall effects beyond clamping
# - equal keys are physically equal; id gives a readout convention

# Lineage
# - The C++ Gist gave particle, radius, and threshold-readout imagery.
# - Bead and gravity sort gave the encode-then-relax pattern.
# - Spaghetti sort warned that readout is not free.
# - Physical Sorting papers gave record linkage and physical readout.
# - Centrifugation literature gave density contrast, viscosity, and gradient.
# - microGPT gave the single-file discipline.

# References
#
# MicroGPT [Forms][Superb]
# The form reference: one file, small top docstring, local comments, and a complete model living in readable code.
#   https://karpathy.github.io/2026/02/12/microgpt/
#   https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
#
# C++ Centrifuge Sort [Materials][Superb]
# The material reference: particle, radius, and threshold-readout imagery; the fake mass drift is explicitly refused.
#   https://gist.github.com/zgoethel/ac3a1b78799582e1c5dd129b9545ea30
#
# Physical sorting theory [Materials][Superb]
# The theory reference: encode -> physical transform -> readout, plus the record-linkage lesson.
# It is the reason particles keep id, and the reason readout is explicit rather than treated as free.
#   https://www.niallmurphy.me/papers/MNWHDDBW2006c.pdf
#   https://dna.hamilton.ie/assets/dw/MNWHDDBW2008p.pdf
#
# Natural sorting family [Materials][Useful]
# Bead and gravity sort give the encode-then-relax pattern and warn that space and representation cost matter.
#   https://www.cs.auckland.ac.nz/research/groups/CDMTCS/researchreports/171joshua.pdf
#   https://turing.iem.thm.de/routeplanning/umc/asg2/umcasg2.pdf
#
# Rainbow sort [Materials][Tentative]
# Useful as analogy for physical-coordinate sorting, less central to the centrifuge model.
#   https://turing.iem.thm.de/routeplanning/rainbowSort/rainbowSortRR.pdf
#
# Centrifugation physics [Materials][Useful]
# The physics reference: density contrast, viscosity, radius, sedimentation, and density gradients.
#   https://www.horiba.com/int/scientific/technologies/centrifugal-sedimentation/
#   https://www.sigmaaldrich.com/US/en/technical-documents/technical-article/protein-biology/protein-pulldown/centrifugation-basics
#
# Forum pressure tests [Materials][Useful]
# The criticism reference: complexity model, readout cost, total-order doubts, and parallelism caveats.
#   https://stackoverflow.com/questions/41584581/sorting-in-computer-science-vs-sorting-in-the-real-world
#   https://www.reddit.com/r/algorithms/comments/5nc3j1/sorting_in_programming_vs_sorting_in_the_real/
#   https://www.personal.kent.edu/~rmuhamma/Algorithms/MyAlgorithms/Sorting/bubbleSort.htm
