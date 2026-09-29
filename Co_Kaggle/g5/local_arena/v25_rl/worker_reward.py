"""Worker-only v25 reward computed from one before/after turn transition."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

CROP_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_PRODUCT_NAMES = ("MILK", "EGG", "WOOL")
PRODUCTS = CROP_PRODUCTS + ANIMAL_PRODUCT_NAMES
ANIMAL_PRODUCTS = {"COW": "MILK", "GOOSE": "EGG", "SHEEP": "WOOL"}

def _zero_crop_counts() -> dict[str, float]:
    return {name: 0.0 for name in CROP_PRODUCTS}

def _zero_animal_product_counts() -> dict[str, float]:
    return {name: 0.0 for name in ANIMAL_PRODUCT_NAMES}

def _zero_product_counts() -> dict[str, float]:
    return {name: 0.0 for name in PRODUCTS}
OPS = (
    "PLANT", "DIG", "HARVEST", "WATER", "FERTILIZE", "FEED", "CARE",
    "COLLECT_FERTILIZER", "BUILD_COOP", "BUILD_PASTURE", "PLACE_ANIMAL",
    "PICKUP", "DELIVER", "PASS",
)
ITEMS = (
    "", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "COW", "SHEEP", "GOOSE", "MILK", "EGG", "WOOL", "FERTILIZER",
)
MOVE_ACTIONS = {"NORTH", "SOUTH", "EAST", "WEST"}

# Crop products keep the original lifecycle value.
PRODUCT_VALUE = 8.0
PRODUCT_GENERATED_REWARD = 2.0
PRODUCT_HARVESTED_REWARD = 2.0
PRODUCT_DELIVERED_REWARD = 4.0
assert PRODUCT_GENERATED_REWARD + PRODUCT_HARVESTED_REWARD + PRODUCT_DELIVERED_REWARD == PRODUCT_VALUE

# Animal products have a longer, lower-throughput production chain
# (build/place/feed/care/harvest/deliver). The observed v4 run produced roughly
# two orders of magnitude fewer animal units than crop units, so animal output
# uses a much larger lifecycle value while crop rewards remain unchanged.
ANIMAL_PRODUCT_VALUE = 128.0
ANIMAL_PRODUCT_GENERATED_REWARD = 16.0
ANIMAL_PRODUCT_HARVESTED_REWARD = 16.0
ANIMAL_PRODUCT_DELIVERED_REWARD = 96.0
assert (
    ANIMAL_PRODUCT_GENERATED_REWARD
    + ANIMAL_PRODUCT_HARVESTED_REWARD
    + ANIMAL_PRODUCT_DELIVERED_REWARD
    == ANIMAL_PRODUCT_VALUE
)

ANIMAL_ESCAPE_PENALTY = -100.0
# Weed is a hard operational failure: scheduler guards should normally prevent
# it, and the remaining transition penalty is intentionally large enough that
# any uncovered failure dominates routine worker shaping.
CROP_TO_WEED_PENALTY = -256.0
CROP_DEATH_PENALTY = -128.0
LOST_HARVESTABLE_UNIT_PENALTY = -32.0

SUCCESSFUL_PLANT_REWARD = 1.0
# Planner execution and PASS preference now live in per-subdecision actor
# advantages. Keep turn reward free of those labels so the critic learns only
# environment outcomes and one worker's choice cannot smear credit over every
# other worker assignment made in the same turn.
PLANNED_PLANT_REWARD = 0.0
BUILD_STRUCTURE_REWARD = 0.5
PLACE_ANIMAL_REWARD = 1.0
PLANNED_PLACE_ANIMAL_REWARD = 0.0
ROUTE_PROGRESS_REWARD = 0.05
AVOIDABLE_PASS_PENALTY = 0.0
EFFECTIVE_CARE_REWARD = 3.0
EFFECTIVE_FERTILIZE_REWARD = 1.0
COLLECT_FERTILIZER_REWARD = 1.0
NORMAL_FEED_REWARD = 6.0
NORMAL_WATER_REWARD = 1.0
CRITICAL_FEED_REWARD = 2.0
HEALTHY_ANIMAL_DAY_REWARD = 4.0
CRITICAL_WATER_REWARD = 16.0


def _positions(obs) -> list[tuple[int, int]]:
    farm = obs["farms"][obs["player"]]
    return [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]


def _tile(obs, p):
    if p is None:
        return None
    farm = obs["farms"][obs["player"]]
    x, y = p
    return farm["tiles"][y][x]


def _worker_actions(action) -> list[list[Any]]:
    if not isinstance(action, dict):
        return []
    farmer = action.get("farmer") or ["PASS"]
    return [farmer] + list(action.get("hands") or [])


def _same_product(tile_before, tile_after) -> str | None:
    if not isinstance(tile_before, dict) or not isinstance(tile_after, dict):
        return None
    if tile_before.get("kind") == "PLANT" and tile_after.get("kind") == "PLANT":
        a, b = tile_before.get("crop"), tile_after.get("crop")
        return a if a == b and a in PRODUCTS else None
    aa, ab = tile_before.get("animal"), tile_after.get("animal")
    if aa and aa == ab:
        return ANIMAL_PRODUCTS.get(aa)
    return None


def _crop_decay_start_step(executor, tile) -> int | None:
    raw_value = tile.get("max_lifespan_step", -1)
    raw = int(raw_value if raw_value is not None else -1)
    if raw >= 0:
        return raw
    crop = tile.get("crop")
    spec = executor.CROPS.get(crop)
    if not spec:
        return None
    events = tuple(spec[2] or ())
    if not events:
        return None
    final_age = max(int(age) for age, _units in events)
    planted = int(tile.get("planted_day", 0) or 0)
    return (planted + final_age + 1) * 24


def _classify_crop_to_weed(executor, before, tile, day_rolled: bool) -> str:
    """Best-evidence cause for a PLANT -> WEED transition."""
    if (
        day_rolled
        and not bool(tile.get("watered_today"))
        and int(tile.get("consecutive_unwatered", 0) or 0) >= 1
    ):
        return "unwatered"

    start = _crop_decay_start_step(executor, tile)
    now = int(before["day"]) * 24 + int(before["hour"])
    if start is not None and now + 1 >= start:
        return "decay"

    return "other"


@dataclass
class RewardBreakdown:
    reward: float = 0.0
    animals_escaped: int = 0
    crops_to_weed: int = 0
    crops_to_weed_unwatered: int = 0
    crops_to_weed_decay: int = 0
    crops_to_weed_other: int = 0
    crops_died: int = 0
    lost_harvestable_units: float = 0.0

    # Existing reward-stage totals.
    products_generated: float = 0.0
    products_harvested: float = 0.0
    products_delivered: float = 0.0

    # Explicit production-pipeline measurements.
    seeds_planted_total: int = 0
    seeds_planted_by_crop: dict[str, float] = field(default_factory=_zero_crop_counts)
    crop_harvest_events_total: int = 0
    crop_harvest_events_by_crop: dict[str, float] = field(default_factory=_zero_crop_counts)
    crop_units_harvested_total: float = 0.0
    crop_units_harvested_by_crop: dict[str, float] = field(default_factory=_zero_crop_counts)
    animal_product_units_generated_total: float = 0.0
    animal_product_units_generated_by_product: dict[str, float] = field(default_factory=_zero_animal_product_counts)
    animal_product_units_harvested_total: float = 0.0
    animal_product_units_harvested_by_product: dict[str, float] = field(default_factory=_zero_animal_product_counts)
    animal_product_units_moved_to_shed_total: float = 0.0
    product_units_moved_to_shed_total: float = 0.0
    product_units_moved_to_shed_by_product: dict[str, float] = field(default_factory=_zero_product_counts)

    plants_created: int = 0
    structures_built: int = 0
    animals_placed: int = 0
    effective_care: int = 0
    effective_fertilize: int = 0
    fertilizer_collected: int = 0
    normal_feed: int = 0
    critical_feed: int = 0
    healthy_animal_days: int = 0
    normal_water: int = 0
    critical_water: int = 0
    planned_plants_completed: int = 0
    planned_animals_placed: int = 0
    route_progress_steps: int = 0
    avoidable_passes: int = 0

    def add(self, other: "RewardBreakdown") -> None:
        for name in self.__dataclass_fields__:
            current = getattr(self, name)
            incoming = getattr(other, name)
            if isinstance(current, dict):
                for key, value in incoming.items():
                    current[key] = current.get(key, 0.0) + float(value)
            else:
                setattr(self, name, current + incoming)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            out[name] = dict(value) if isinstance(value, dict) else value
        return out


def compute_worker_reward(executor, before, worker_action, after) -> RewardBreakdown:
    """Calculate reward for worker execution only.

    worker_action contains farmer/hands plus optional reward-only delivery
    provenance metadata. Market orders are intentionally not accepted by this
    API, so buying/selling/hiring/land actions cannot directly enter worker
    reward attribution.

    The before/after observations still come from the real environment turn,
    which may include frozen market actions and day refresh. The emitted worker
    actions are used to disambiguate worker-controlled effects that the final
    next-turn observation can otherwise hide.
    """
    out = RewardBreakdown()
    player = int(before["player"])
    before_farm = before["farms"][player]
    after_farm = after["farms"][player]
    if set(worker_action) - {
        "farmer", "hands", "_delivery_credit", "_plan_credit",
        "_route_progress", "_avoidable_pass",
    }:
        raise ValueError("worker reward received unsupported action fields")
    actions = _worker_actions(worker_action)
    before_positions = _positions(before)
    before_invs = list(before["private"]["inventories"])
    after_invs = list(after["private"]["inventories"])
    delivery_credit = worker_action.get("_delivery_credit") or []
    plan_credit = worker_action.get("_plan_credit") or []
    route_progress = worker_action.get("_route_progress") or []
    avoidable_pass = worker_action.get("_avoidable_pass") or []

    for i, action in enumerate(actions):
        if i < len(route_progress) and bool(route_progress[i]):
            out.route_progress_steps += 1
            out.reward += ROUTE_PROGRESS_REWARD
        if (
            action
            and action[0] == "PASS"
            and i < len(avoidable_pass)
            and bool(avoidable_pass[i])
        ):
            out.avoidable_passes += 1
            out.reward += AVOIDABLE_PASS_PENALTY

    day_rolled = int(after["day"]) != int(before["day"])
    shed_points = {tuple(p) for p in executor.SHED}

    action_at: dict[tuple[int, int], list[tuple[int, list[Any]]]] = defaultdict(list)
    harvested_by_tile: dict[tuple[int, int], float] = defaultdict(float)

    # Action-derived events whose post-state can be obscured by market/day refresh.
    shed_load = sum(int(v) for v in before["private"]["shed"].values())
    shed_capacity_left = max(0, int(getattr(executor, "SHED_CAPACITY", 100)) - shed_load)
    for i, action in enumerate(actions):
        if i >= len(before_positions) or not action:
            continue
        pos = before_positions[i]
        op = action[0]
        action_at[pos].append((i, action))
        tile_before = _tile(before, pos)
        inv_before = before_invs[i] if i < len(before_invs) else {}

        if op == "HARVEST" and isinstance(tile_before, dict):
            units = float(tile_before.get("yield_units", 0) or 0)
            legal_harvest = units > 0

            if tile_before.get("kind") == "PLANT":
                crop = tile_before.get("crop")
                age = int(before["day"]) - int(tile_before.get("planted_day", before["day"]))
                first_yield_age = int(
                    getattr(executor, "CROP_FIRST_YIELD_DAY", {}).get(crop, 10**9)
                )
                # Match the engine's legal HARVEST threshold. Do not infer
                # legality from the planner's nominal/max-yield schedule.
                legal_harvest = legal_harvest and age >= first_yield_age

            if legal_harvest:
                harvested_by_tile[pos] += units
                out.products_harvested += units

                if tile_before.get("kind") == "PLANT":
                    out.reward += PRODUCT_HARVESTED_REWARD * units
                    crop = tile_before.get("crop")
                    if crop in CROP_PRODUCTS:
                        out.crop_harvest_events_total += 1
                        out.crop_harvest_events_by_crop[crop] += 1
                        out.crop_units_harvested_total += units
                        out.crop_units_harvested_by_crop[crop] += units
                elif tile_before.get("animal"):
                    product = ANIMAL_PRODUCTS.get(tile_before.get("animal"))
                    if product in ANIMAL_PRODUCT_NAMES:
                        out.reward += ANIMAL_PRODUCT_HARVESTED_REWARD * units
                        out.animal_product_units_harvested_total += units
                        out.animal_product_units_harvested_by_product[product] += units

        elif op == "FERTILIZE" and isinstance(tile_before, dict) and tile_before.get("kind") == "PLANT":
            has_resource = float(inv_before.get("FERTILIZER", 0) or 0) > 0
            after_tile = _tile(after, pos)
            effective = (
                has_resource
                and isinstance(after_tile, dict)
                and after_tile.get("kind") == "PLANT"
                and after_tile.get("crop") == tile_before.get("crop")
                and int(after_tile.get("fertilized_until_day", -1))
                    > int(tile_before.get("fertilized_until_day", -1))
            )
            if effective:
                out.effective_fertilize += 1
                out.reward += EFFECTIVE_FERTILIZE_REWARD

        elif op == "COLLECT_FERTILIZER" and isinstance(tile_before, dict) and tile_before.get("animal"):
            if bool(tile_before.get("fertilizer_available")):
                # At day rollover a new fertilizer unit can immediately become
                # available again, so the valid precondition is the stable signal.
                out.fertilizer_collected += 1
                out.reward += COLLECT_FERTILIZER_REWARD

        elif (
            op == "PLACE" and len(action) >= 2 and action[1] in PRODUCTS
            and pos in shed_points
        ):
            product = action[1]
            requested = int(action[2]) if len(action) >= 3 else 1
            available = int(inv_before.get(product, 0) or 0)
            # Delivery reward is for newly produced logistics, not circulation.
            # WHEAT can be PICKUP'd from the shed for feeding, so without
            # provenance the same units could be repeatedly PICKUP -> PLACE'd
            # for positive reward. If provenance metadata is present, cap every
            # product by it. Without metadata, be conservative for WHEAT and
            # grant no delivery credit; other products cannot currently be
            # PICKUP-generated by WorkerPolicy.
            if i < len(delivery_credit) and isinstance(delivery_credit[i], dict):
                provenance = max(0, int(delivery_credit[i].get(product, 0) or 0))
            elif product == "WHEAT":
                provenance = 0
            else:
                provenance = available
            accepted = max(
                0, min(requested, available, shed_capacity_left, provenance)
            )
            if accepted:
                out.products_delivered += accepted
                out.product_units_moved_to_shed_total += accepted
                out.product_units_moved_to_shed_by_product[product] += accepted
                if product in ANIMAL_PRODUCT_NAMES:
                    out.animal_product_units_moved_to_shed_total += accepted
                    delivery_reward = ANIMAL_PRODUCT_DELIVERED_REWARD
                else:
                    delivery_reward = PRODUCT_DELIVERED_REWARD
                out.reward += delivery_reward * accepted
                shed_capacity_left -= accepted

    # Tile-level before/after events.
    h = len(before_farm["tiles"])
    w = len(before_farm["tiles"][0]) if h else 0
    for y in range(h):
        for x in range(w):
            p = (x, y)
            bt = before_farm["tiles"][y][x]
            at = after_farm["tiles"][y][x]
            tile_actions = [a for _, a in action_at.get(p, [])]
            ops_here = {a[0] for a in tile_actions if a}

            # Catastrophic animal loss.
            if isinstance(bt, dict) and bt.get("animal"):
                same_animal = isinstance(at, dict) and at.get("animal") == bt.get("animal")
                if not same_animal:
                    out.animals_escaped += 1
                    lost = float(bt.get("yield_units", 0) or 0)
                    out.lost_harvestable_units += lost
                    out.reward += ANIMAL_ESCAPE_PENALTY + LOST_HARVESTABLE_UNIT_PENALTY * lost

            # Keep the historical total, but classify the cause. A crop can
            # become WEED through missed watering or through normal lifespan
            # decay after harvestable yield is left on the tile.
            if isinstance(bt, dict) and bt.get("kind") == "PLANT":
                if isinstance(at, dict) and at.get("kind") == "WEED":
                    out.crops_to_weed += 1
                    cause = _classify_crop_to_weed(
                        executor, before, bt, day_rolled
                    )
                    if cause == "unwatered":
                        out.crops_to_weed_unwatered += 1
                    elif cause == "decay":
                        out.crops_to_weed_decay += 1
                    else:
                        out.crops_to_weed_other += 1
                    lost = float(bt.get("yield_units", 0) or 0)
                    out.lost_harvestable_units += lost
                    out.reward += CROP_TO_WEED_PENALTY + LOST_HARVESTABLE_UNIT_PENALTY * lost
                elif at is None and "HARVEST" not in ops_here:
                    # DIG of a fully exhausted crop is valid cleanup, not crop
                    # death. Destroying a still-productive plant remains a heavy
                    # worker-efficiency failure.
                    crop = bt.get("crop")
                    spec = executor.CROPS.get(crop)
                    age = int(before["day"]) - int(bt.get("planted_day", before["day"]))
                    useful_age = (
                        max(a for a, _ in spec[2])
                        if spec and crop in ("TOMATO", "STRAWBERRY")
                        else int(spec[3]) if spec else -1
                    )
                    exhausted_cleanup = (
                        "DIG" in ops_here
                        and float(bt.get("yield_units", 0) or 0) <= 0
                        and age > useful_age
                    )
                    if not exhausted_cleanup:
                        out.crops_died += 1
                        lost = float(bt.get("yield_units", 0) or 0)
                        out.lost_harvestable_units += lost
                        out.reward += CROP_DEATH_PENALTY + LOST_HARVESTABLE_UNIT_PENALTY * lost

            # Planting on the last hour can immediately refresh into a weed;
            # count that as the same catastrophic failure rather than a random weed.
            if bt is None and isinstance(at, dict) and at.get("kind") == "WEED" and "PLANT" in ops_here:
                # The seed was still planted successfully; the day refresh then
                # killed it immediately. Count both the planting event and the
                # worker-efficiency failure.
                planted_crop = next(
                    (
                        action[1]
                        for action in tile_actions
                        if action and action[0] == "PLANT" and len(action) >= 2
                    ),
                    None,
                )
                if planted_crop in CROP_PRODUCTS:
                    out.seeds_planted_total += 1
                    out.seeds_planted_by_crop[planted_crop] += 1
                out.crops_to_weed += 1
                out.crops_to_weed_unwatered += 1
                out.reward += CROP_TO_WEED_PENALTY

            # Successful capacity creation.
            if bt is None and isinstance(at, dict) and at.get("kind") == "PLANT":
                out.plants_created += 1
                crop = at.get("crop")
                if crop in CROP_PRODUCTS:
                    out.seeds_planted_total += 1
                    out.seeds_planted_by_crop[crop] += 1
                out.reward += SUCCESSFUL_PLANT_REWARD
                for worker_i, _action in action_at.get(p, []):
                    credit = plan_credit[worker_i] if worker_i < len(plan_credit) else {}
                    if (
                        isinstance(credit, dict)
                        and credit.get("op") == "PLANT"
                        and credit.get("item") == crop
                    ):
                        out.planned_plants_completed += 1
                        out.reward += PLANNED_PLANT_REWARD
                        break
            if bt is None and isinstance(at, dict) and at.get("kind") in ("COOP", "PASTURE"):
                if not at.get("animal"):
                    out.structures_built += 1
                    out.reward += BUILD_STRUCTURE_REWARD
            if (
                isinstance(bt, dict) and bt.get("kind") in ("COOP", "PASTURE")
                and not bt.get("animal")
                and isinstance(at, dict) and at.get("animal")
            ):
                out.animals_placed += 1
                out.reward += PLACE_ANIMAL_REWARD
                animal = at.get("animal")
                for worker_i, _action in action_at.get(p, []):
                    credit = plan_credit[worker_i] if worker_i < len(plan_credit) else {}
                    if (
                        isinstance(credit, dict)
                        and credit.get("op") == "PLACE_ANIMAL"
                        and credit.get("item") == animal
                    ):
                        out.planned_animals_placed += 1
                        out.reward += PLANNED_PLACE_ANIMAL_REWARD
                        break

            # Water/feed are effective only when the same asset survives.
            if isinstance(bt, dict) and bt.get("kind") == "PLANT" and "WATER" in ops_here:
                same = isinstance(at, dict) and at.get("kind") == "PLANT" and at.get("crop") == bt.get("crop")
                success = same and (
                    bool(at.get("watered_today"))
                    or (day_rolled and int(at.get("consecutive_unwatered", 99)) == 0)
                )
                if success:
                    if int(bt.get("consecutive_unwatered", 0) or 0) >= 1:
                        out.critical_water += 1
                        out.reward += CRITICAL_WATER_REWARD
                    else:
                        out.normal_water += 1
                        out.reward += NORMAL_WATER_REWARD

            if isinstance(bt, dict) and bt.get("animal") and "FEED" in ops_here:
                same = isinstance(at, dict) and at.get("animal") == bt.get("animal")
                success = same and (
                    bool(at.get("fed_today"))
                    or (day_rolled and int(at.get("consecutive_unfed", 99)) == 0)
                )
                if success:
                    if int(bt.get("consecutive_unfed", 0) or 0) >= 1:
                        out.critical_feed += 1
                        out.reward += CRITICAL_FEED_REWARD
                    else:
                        out.normal_feed += 1
                        out.reward += NORMAL_FEED_REWARD

            # End-of-day animal maintenance credit. CARE is only economically
            # effective when the animal finishes the day both fed and cared.
            # Credit that outcome at rollover rather than requiring FEED and CARE
            # to happen in the same turn. This correctly handles CARE early in
            # the day followed by FEED later, and avoids rewarding CARE that was
            # never paired with feeding.
            if day_rolled and isinstance(bt, dict) and bt.get("animal"):
                same = isinstance(at, dict) and at.get("animal") == bt.get("animal")
                fed_for_day = bool(bt.get("fed_today"))
                cared_for_day = bool(bt.get("cared_today"))
                if same and "FEED" in ops_here:
                    fed_for_day = (
                        fed_for_day
                        or int(at.get("consecutive_unfed", 99)) == 0
                    )
                if same and "CARE" in ops_here:
                    cared_for_day = True

                if same and fed_for_day:
                    out.healthy_animal_days += 1
                    out.reward += HEALTHY_ANIMAL_DAY_REWARD

                if same and fed_for_day and cared_for_day:
                    out.effective_care += 1
                    out.reward += EFFECTIVE_CARE_REWARD

            # Real output generated this turn.  If the same tile was harvested,
            # add harvested units back before differencing so production after a
            # harvest (e.g. at day refresh) is still visible.
            product = _same_product(bt, at)
            if product:
                before_yield = float(bt.get("yield_units", 0) or 0)
                after_yield = float(at.get("yield_units", 0) or 0)
                generated = max(0.0, after_yield + harvested_by_tile.get(p, 0.0) - before_yield)
                if generated:
                    out.products_generated += generated
                    if bt.get("animal"):
                        animal_product = ANIMAL_PRODUCTS.get(bt.get("animal"))
                        if animal_product in ANIMAL_PRODUCT_NAMES:
                            out.animal_product_units_generated_total += generated
                            out.animal_product_units_generated_by_product[animal_product] += generated
                            out.reward += ANIMAL_PRODUCT_GENERATED_REWARD * generated
                    else:
                        out.reward += PRODUCT_GENERATED_REWARD * generated

    return out
