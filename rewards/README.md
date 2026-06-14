# rewards/ — GRPO reward definitions (operator, not shipped)

A GRPO reward is `f(idr: str) -> float` over the decoded IDR residue string. There are two homes:

- **Generic built-ins** ship in the library: `idiom.train.grpo.rewards` (`fraction_proline`,
  `fraction_alanine`, plus the `length` / `entropy` penalty terms and the registry).
- **This dir** (not in the wheel) holds:
  - **`example_rewards.py`** — copyable examples of custom rewards; point `reward.module` at it.
  - **`protgps/`** — the vendored ProtGPS localization model, wrapped by a `protgps` reward
    (operator-registered) for compartment-specific RL.

Add a reward with `@register_reward("name")`, then select it in config:
```
idiom_grpo init_from=... reward.module=rewards/example_rewards.py reward.name=aromatic_fraction
```
