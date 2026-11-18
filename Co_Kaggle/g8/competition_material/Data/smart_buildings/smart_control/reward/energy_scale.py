"""Energy-normalisation scale: max_*_rate = c * K, K = air handlers.

The denominators scale with the air handler count rather than being flat. A
flat value would give a per-AHU allowance that varies with building size for no
physical reason, so identical draw would score differently depending on how many
air handlers a building happens to have -- and it would over-charge any policy
that runs the plant relative to one that does not.

The configs now carry `c * K` directly, so nothing at runtime needs to apply
this -- `load_hybrid_environment` picks it up like any other gin value. These
helpers exist to *check* that, and for code that builds a reward function
without going through a shipped config.

Note: overriding at runtime must be done on the constructed object.
`gin.bind_parameter` does not work -- `load_hybrid_environment` re-parses the
config and silently discards anything bound beforehand.
"""

# The denominators are set so that the REALISED contribution of energy matches
# the weight the reward declares for it.
#
# The reward weights comfort 0.6 against energy plus carbon 0.4, but those
# weights only describe the outcome if the normalisation puts the two terms on
# a comparable footing. Scaling the denominators up shrinks the energy term and
# lets comfort dominate; scaling them down does the reverse. At these values a
# controller near the frontier loses roughly 60% of its reward to comfort and
# 40% to fuel, which is what the weights say should happen.
#
# The alternative rule -- setting the scale to the observed peak draw -- makes
# the normalised term land near 1.0 at maximum load, but that is a plotting
# convention rather than a statement about the reward, and it left energy doing
# about two thirds of its declared work.
#
# Both fuels are derived by the same rule deliberately. Deriving them
# differently would bias the reward between electricity and gas, and a policy
# can trade one for the other.
#
# There is a floor below which this cannot go: make energy expensive enough and
# the score-maximising response is to stop conditioning the building. That
# reappears only below roughly a third of these values, so the margin is wide.
#
# These are normalisation scales, not caps. Draw above them is not clipped.
C_ELECTRICITY_PER_AHU = 80000.0
C_NATURAL_GAS_PER_AHU = 205000.0


def n_air_handlers(env) -> int:
    """K, counted from the constructed environment.

    Counted rather than inferred from the action width. The continuous action
    is `3K + 2 + 2V` for K air handlers and V VAV boxes, so its width alone
    does not determine K, and any arithmetic that assumes a fixed layout will
    return a plausible but wrong answer when the action space changes.
    """
    return len(env.building.simulator._hvac.air_handler._ahus)


def expected_rates(env) -> tuple[float, float]:
    """The (electricity, gas) denominators this env should be using."""
    k = n_air_handlers(env)
    return C_ELECTRICITY_PER_AHU * k, C_NATURAL_GAS_PER_AHU * k


def check_energy_scale(env) -> None:
    """Raise if the env's live denominators disagree with c * K."""
    want_e, want_g = expected_rates(env)
    rf = env.reward_function
    got_e = float(rf._max_electricity_rate)
    got_g = float(rf._max_natural_gas_rate)
    if got_e != want_e or got_g != want_g:
        raise ValueError(
            f"energy scale mismatch for K={n_air_handlers(env)}: "
            f"electricity {got_e:,.0f} (want {want_e:,.0f}), "
            f"gas {got_g:,.0f} (want {want_g:,.0f})"
        )


def apply_energy_scale(env) -> tuple[float, float]:
    """Force c * K onto a constructed env. Returns the applied rates."""
    want_e, want_g = expected_rates(env)
    rf = env.reward_function
    rf._max_electricity_rate = want_e
    rf._max_natural_gas_rate = want_g
    check_energy_scale(env)
    return want_e, want_g
