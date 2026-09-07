"""
Verification script for Target Returns Clipping issue.

Problem: Code clipsGROUND TRUTH target returns if normalize_returns=False
and value_clip_limit is set.

Location: distributional_ppo.py:9191-9200, 10258-10265

Code:
    if (not self.normalize_returns) and (self._value_clip_limit_unscaled is not None):
        limit_unscaled = float(self._value_clip_limit_unscaled)
        target_returns_raw = torch.clamp(
            target_returns_raw,
            min=-limit_unscaled,
            max=limit_unscaled,
        )

This clips the TARGET (ground truth), not the PREDICTION changes!

This would be catastrophic if enabled:
- If limit = 0.2 (typical PPO epsilon)
- And actual return = 1.0 (100% profit)
- Target would be clipped to 0.2
- Model can't learn beyond 20% returns!
"""

import yaml
from pathlib import Path


def check_config_files():
    """Check all config files for value_clip_limit setting."""
    print("=" * 80)
    print("Checking Config Files for value_clip_limit")
    print("=" * 80)
    print()

    configs_dir = Path("configs")
    config_files = list(configs_dir.glob("*.yaml"))

    results = []

    for config_file in config_files:
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Check for value_clip_limit in various locations
            clip_limit = None

            # Check in arch_params.critic
            if isinstance(config, dict):
                arch_params = config.get("arch_params", {})
                if isinstance(arch_params, dict):
                    critic = arch_params.get("critic", {})
                    if isinstance(critic, dict):
                        clip_limit = critic.get("value_clip_limit")

                # Also check model.params (if exists)
                model = config.get("model", {})
                if isinstance(model, dict):
                    params = model.get("params", {})
                    if isinstance(params, dict):
                        # value_clip_limit might be here too
                        if "value_clip_limit" in params:
                            clip_limit = params["value_clip_limit"]

            results.append((config_file.name, clip_limit))

        except Exception as e:
            results.append((config_file.name, f"ERROR: {e}"))

    # Print results
    for filename, clip_limit in results:
        if clip_limit is None or clip_limit == "null":
            status = "[SAFE] null (disabled)"
        elif isinstance(clip_limit, (int, float)):
            if clip_limit <= 0:
                status = f"[SAFE] {clip_limit} (disabled: <= 0)"
            else:
                status = f"[DANGEROUS] {clip_limit} (ACTIVE!)"
        else:
            status = f"[UNKNOWN] {clip_limit}"

        print(f"{filename:40s}: {status}")

    print()
    return results


def analyze_code():
    """Analyze how value_clip_limit is used in code."""
    print("=" * 80)
    print("Code Analysis")
    print("=" * 80)
    print()

    print("HOW value_clip_limit IS SET:")
    print("  Location: distributional_ppo.py:6798-6809")
    print("  Source:   self.policy.value_clip_limit")
    print("  Condition: Only active if normalize_returns=False")
    print()

    print("WHERE IT'S USED TO CLIP TARGETS:")
    print("  1. distributional_ppo.py:9191-9200 (rollout collection)")
    print("  2. distributional_ppo.py:10258-10265 (train step)")
    print()

    print("WHAT IT DOES:")
    print("  Clips TARGET returns (ground truth) to [-limit, +limit]")
    print("  NOT prediction changes (which would be correct PPO VF clipping)")
    print()

    print("WHY THIS IS WRONG:")
    print("  PPO value clipping should clip PREDICTION CHANGES:")
    print("    V_clipped = V_old + clip(V_new - V_old, -epsilon, +epsilon)")
    print()
    print("  NOT the target itself:")
    print("    target_clipped = clip(target, -epsilon, +epsilon)  <-- WRONG!")
    print()

    print("EXAMPLE OF CATASTROPHIC FAILURE:")
    print("  Scenario: Model achieves 100% profit on a trade")
    print("  - Actual return: 1.0 (100%)")
    print("  - value_clip_limit: 0.2 (typical PPO epsilon)")
    print("  - Clipped target: 0.2 (20%)")
    print("  - Model learns: Max possible return is 0.2")
    print("  - Result: Model can NEVER learn returns > 20%!")
    print()


def check_current_configs():
    """Check if the bug is currently active in any configs."""
    print("=" * 80)
    print("Is The Bug Currently Active?")
    print("=" * 80)
    print()

    results = check_config_files()

    # Check if any config has non-null, positive value_clip_limit
    active_configs = []
    for filename, clip_limit in results:
        if isinstance(clip_limit, (int, float)) and clip_limit > 0:
            active_configs.append((filename, clip_limit))

    if active_configs:
        print("[DANGER] BUG IS ACTIVE in the following configs:")
        for filename, clip_limit in active_configs:
            print(f"  - {filename}: value_clip_limit={clip_limit}")
        print()
        print("RECOMMENDATION: Set value_clip_limit to null or remove it!")
    else:
        print("[SAFE] Bug is NOT currently active.")
        print("  All configs have value_clip_limit=null or undefined.")
        print()
        print("However, the buggy code still exists and could be activated")
        print("if someone sets value_clip_limit without understanding the bug.")

    print()

    return len(active_configs) > 0


def main():
    print("\n")
    print("=" * 80)
    print("Target Returns Clipping Verification")
    print("=" * 80)
    print("\n")

    analyze_code()
    is_active = check_current_configs()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if is_active:
        print("[CRITICAL] BUG IS ACTIVE!")
        print("  Target returns are being clipped, preventing model from learning large returns.")
        print("  IMMEDIATE ACTION REQUIRED: Disable value_clip_limit in all configs!")
    else:
        print("[STATUS] Bug exists in code but is NOT currently active.")
        print("  value_clip_limit is null/disabled in all configs.")
        print()
        print("RECOMMENDATION:")
        print("  1. Document this bug clearly (add warning comments)")
        print("  2. Consider removing the buggy code entirely")
        print("  3. OR: Rename to something more explicit like 'target_clip_limit'")
        print("     to make it clear this clips TARGETS, not prediction changes")
        print()
        print("For now, system is SAFE as long as value_clip_limit remains null.")


if __name__ == "__main__":
    main()
