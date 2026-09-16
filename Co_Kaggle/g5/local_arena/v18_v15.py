"""Kaggriculture v15: demand-forecast crop planning and compact livestock care.

Forecast crop sale values from market inventory, visible crops on both farms,
current shops, and expected future shop demand. Allocate up to 42 crop plots
across NW/NE without replacing healthy growing crops. Service each livestock
tile before moving on, use automatic daily inventory delivery, and fertilize
ongoing crops near their production windows. Uses only public observations
and the player's own inventory; no external files or dependencies.
"""


PASS = ["PASS"]
MAX_MARKET_ORDERS = 10
FIRST_LAND_COST = 1000
EXPAND_LAND = True
LAND_BUY_CUTOFF = 14
LAND_CASH_BUFFER = 800
BASE_HANDS = 7
EXPANDED_HANDS = 10
FINAL_PLANT_HOUR = 18
SEED_BUY_CUTOFF = 8
LIQUIDATION_DAY = 27
FINAL_CARE_DAY = 28
FEED_TARGET = 21
FEED_REORDER = 10
FERTILIZER_RESERVE = 4
FERTILIZER_LOAD = 6
WHEAT_STOCKPILE_DAY = 8
WHEAT_STOCKPILE_TARGET = 25
WHEAT_STOCKPILE_REORDER = 12
ADAPT_CARROT_PRICE = 45
ADAPT_STRAWBERRY_PRICE = 150
ADAPT_TOMATO_PRICE = 80


ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP", "product": "EGG"},
    "COW": {"cost": 400, "structure": "PASTURE", "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "product": "WOOL"},
}


# Three compact cow/sheep routes form the core herd; a two-goose route is
# activated only after northeast land unlocks.
BASE_ANIMAL_ROUTE_COUNT = 3
ANIMAL_ROUTES = (
    ((4, 4, "COW"), (3, 3, "SHEEP")),
    ((3, 4, "COW"), (2, 3, "SHEEP")),
    ((4, 3, "COW"), (2, 2, "SHEEP")),
    ((5, 3, "GOOSE"), (5, 2, "GOOSE")),
)
ANIMAL_SLOTS = tuple(slot for route in ANIMAL_ROUTES for slot in route)

CROPS = {
    "WHEAT": {
        "seed_cost": 10,
        "base_price": 25,
        "first_yield_day": 2,
        "harvest_day": 4,
        "last_plant_day": 25,
        "ongoing": False,
        "final_age": 4,
    },
    "CARROT": {
        "seed_cost": 20,
        "base_price": 35,
        "first_yield_day": 2,
        "harvest_day": 3,
        "last_plant_day": 25,
        "ongoing": False,
        "final_age": 3,
    },
    "TOMATO": {
        "seed_cost": 50,
        "base_price": 60,
        "first_yield_day": 8,
        "harvest_day": 8,
        "last_plant_day": 20,
        "ongoing": True,
        "final_age": 11,
    },
    "STRAWBERRY": {
        "seed_cost": 100,
        "base_price": 120,
        "first_yield_day": 10,
        "harvest_day": 10,
        "last_plant_day": 18,
        "ongoing": True,
        "final_age": 16,
    },
    "MELON": {
        "seed_cost": 80,
        "base_price": 250,
        "first_yield_day": 10,
        "harvest_day": 10,
        "last_plant_day": 18,
        "ongoing": False,
        "final_age": 10,
    },
}


# Candidate crop coordinates; the forecast chooses the actual crop.
CROP_SLOTS = (
    (0, 0, "WHEAT"), (1, 0, "STRAWBERRY"), (2, 0, "WHEAT"),
    (3, 0, "WHEAT"), (4, 0, "MELON"),
    (0, 1, "WHEAT"), (1, 1, "WHEAT"), (2, 1, "STRAWBERRY"),
    (3, 1, "MELON"), (4, 1, "TOMATO"),
    (0, 2, "WHEAT"), (1, 2, "WHEAT"), (2, 2, "WHEAT"),
    (3, 2, "STRAWBERRY"), (4, 2, "MELON"),
    (0, 3, "STRAWBERRY"), (1, 3, "MELON"), (2, 3, "WHEAT"),
    (3, 3, "WHEAT"),
    (0, 4, "WHEAT"), (1, 4, "WHEAT"), (2, 4, "TOMATO"),
)


# Nearby northeast plots are extended to the full quadrant below.
_EXTRA_POINTS = (
    (5, 3), (6, 4), (5, 2), (6, 3), (7, 4), (5, 1),
    (6, 2), (7, 3), (8, 4), (5, 0), (6, 1), (7, 2),
)
CROP_SLOTS = CROP_SLOTS + tuple(
    (x, y, "WHEAT" if index < 8 else ("STRAWBERRY" if index < 10 else "TOMATO"))
    for index, (x, y) in enumerate(_EXTRA_POINTS)
)
_ANIMAL_POINTS = {(x, y) for x, y, _ in ANIMAL_SLOTS}
CROP_SLOTS = tuple(slot for slot in CROP_SLOTS if slot[:2] not in _ANIMAL_POINTS)
_EXTRA_RANK = {point: index for index, point in enumerate(_EXTRA_POINTS)}


def _desired_crop(farm,x,y,planned,day):
    return _crop_plan(farm,day).get((x,y),planned)


SELL_RULES = {
    # Fertilizer has no town demand, so holding cannot create recovery.
    "FERTILIZER": (12, 1),
    "EGG": (12, 38),
    "MILK": (5, 90),
    "WOOL": (4, 120),
    "MELON": (6, 155),
    "STRAWBERRY": (4, 75),
    "TOMATO": (6, 35),
    "CARROT": (12, 23),
    "WHEAT": (16, 19),
}


SELL_ORDER = (
    "WOOL", "MILK", "MELON", "STRAWBERRY", "FERTILIZER",
    "EGG", "TOMATO", "CARROT", "WHEAT",
)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _step_toward(position, target):
    x, y = position
    tx, ty = target
    dx = tx - x
    dy = ty - y
    if abs(dx) >= abs(dy) and dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    return PASS


def _shed_tiles(board_size=10):
    half = board_size // 2
    return (
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    )


def _at_shed(position, board_size=10):
    return tuple(position) in _shed_tiles(board_size)


def _nearest_shed(position, board_size=10):
    x, y = position
    return min(
        _shed_tiles(board_size),
        key=lambda point: (abs(point[0] - x) + abs(point[1] - y), point[1], point[0]),
    )


def _unit_inventory(private, unit_index):
    inventories = private.get("inventories", []) or []
    if 0 <= unit_index < len(inventories) and isinstance(inventories[unit_index], dict):
        return inventories[unit_index]
    return {}


def _tile_at(farm, x, y):
    try:
        return farm["tiles"][y][x]
    except (KeyError, IndexError, TypeError):
        return "LOCKED"


def _animal_counts(farm, private):
    counts = {animal: 0 for animal in ANIMALS}
    for row in (farm.get("tiles", []) or []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in counts:
                counts[tile["animal"]] += 1
    shed = private.get("shed", {}) or {}
    for animal in counts:
        counts[animal] += max(0, _safe_int(shed.get(animal), 0))
    for inventory in (private.get("inventories", []) or []):
        if not isinstance(inventory, dict):
            continue
        for animal in counts:
            counts[animal] += max(0, _safe_int(inventory.get(animal), 0))
    return counts


def _route_live_slots(farm, route):
    live = []
    for slot in route:
        x, y, animal = slot
        tile = _tile_at(farm, x, y)
        if isinstance(tile, dict) and tile.get("animal") == animal:
            live.append((slot, tile))
    return live


def _nearest_route_task(position, tasks):
    return min(
        tasks,
        key=lambda task: (
            abs(task[0][0] - position[0]) + abs(task[0][1] - position[1]),
            task[0][1],
            task[0][0],
        ),
    )


def _animal_route_action(farm, private, unit_index, route, day):
    positions = [tuple(farm.get("farmer", (4,4)))] + [tuple(p) for p in farm.get("hands", [])]
    position = positions[unit_index]
    inventory = _unit_inventory(private, unit_index)
    shed = private.get("shed", {})
    live = _route_live_slots(farm, route)
    missing = [slot for slot in route if not any(slot == s for s,t in live)]
    if day == 29 and any(inventory.get(item,0) for item in SELL_RULES):
        return ["DROP"] if _at_shed(position) else _step_toward(position,_nearest_shed(position))
    for x,y,animal in missing:
        if inventory.get(animal,0):
            if position != (x,y):
                return _step_toward(position,(x,y))
            tile = _tile_at(farm,x,y)
            if tile is None:
                return ["BUILD_COOP" if animal == "GOOSE" else "BUILD_PASTURE"]
            if tile.get("kind") == ANIMALS[animal]["structure"] and not tile.get("animal"):
                return ["PLACE",animal]
            return ["DIG"]
    unfed = [(s,t) for s,t in live if not t.get("fed_today") and day <= FINAL_CARE_DAY]
    if unfed and not inventory.get("WHEAT",0):
        if _at_shed(position) and shed.get("WHEAT",0):
            return ["PICKUP","WHEAT",min(len(unfed),shed["WHEAT"])]
        return _step_toward(position,_nearest_shed(position))
    tasks = []
    for slot,tile in live:
        action = None
        if day <= FINAL_CARE_DAY and not tile.get("fed_today"):
            action = ["FEED"]
        elif tile.get("yield_units",0):
            action = ["HARVEST"]
        elif day <= FINAL_CARE_DAY and not tile.get("cared_today"):
            action = ["CARE"]
        elif tile.get("fertilizer_available"):
            action = ["COLLECT_FERTILIZER"]
        if action:
            distance = abs(position[0]-slot[0])+abs(position[1]-slot[1])
            tasks.append((distance,slot[1],slot[0],action))
    if tasks:
        _,y,x,action = min(tasks)
        return action if position == (x,y) else _step_toward(position,(x,y))
    if missing and day <= 18:
        animal = missing[0][2]
        if _at_shed(position) and shed.get(animal,0):
            return ["PICKUP",animal,1]
        return _step_toward(position,_nearest_shed(position))
    # Daily collection automatically banks all carried goods; drop only when
    # already at storage. Crop assignment can use the remainder of the day.
    if _at_shed(position) and any(inventory.get(item,0) for item in SELL_RULES):
        return ["DROP"]
    return None


def _animal_route_is_done(farm, private, unit_index, route, day):
    """Return True when both animals are fully serviced and loads are banked."""
    live = _route_live_slots(farm, route)
    if len(live) != len(route):
        return False
    inventory = _unit_inventory(private, unit_index)
    if any(_safe_int(inventory.get(item), 0) > 0 for item in SELL_RULES):
        return False
    for _, tile in live:
        if day <= FINAL_CARE_DAY and not bool(tile.get("fed_today", False)):
            return False
        if _safe_int(tile.get("yield_units"), 0) > 0:
            return False
        if day <= FINAL_CARE_DAY and not bool(tile.get("cared_today", False)):
            return False
        if bool(tile.get("fertilizer_available", False)):
            return False
    return True


def _crop_task(tile, crop, day, hour):
    data = CROPS[crop]
    if tile is None:
        if day <= data["last_plant_day"] and hour <= FINAL_PLANT_HOUR:
            return 4, ["PLANT", crop]
        return None
    if tile == "LOCKED" or not isinstance(tile, dict):
        return None
    if tile.get("kind") == "WEED":
        return 3, ["DIG"]
    if tile.get("kind") != "PLANT":
        return None

    actual = tile.get("crop")
    actual_data = CROPS.get(actual)
    if actual_data is None:
        return 3, ["DIG"]
    age = day - _safe_int(tile.get("planted_day"), day)
    held = _safe_int(tile.get("yield_units"), 0)
    watered = bool(tile.get("watered_today", False))

    if day >= 28 and held > 0 and age >= actual_data["first_yield_day"]:
        return 0, ["HARVEST"]
    if actual_data["ongoing"]:
        if held >= 4 or (age >= actual_data["final_age"] and held > 0):
            return 0, ["HARVEST"]
        if age > actual_data["final_age"] and held <= 0:
            return 3, ["DIG"]
        if not watered:
            return 1, ["WATER"]
        return None

    if held > 0 and age >= actual_data["harvest_day"]:
        if age == actual_data["harvest_day"] and not watered:
            return 0, ["WATER"]
        return 0, ["HARVEST"]
    if not watered:
        return 1, ["WATER"]
    return None


def _crop_tasks(farm, private, day, hour):
    seeds = {
        crop: _safe_int((private.get("seeds", {}) or {}).get(crop), 0)
        for crop in CROPS
    }
    tasks = []
    for x, y, crop in CROP_SLOTS:
        crop = _desired_crop(farm, x, y, crop, day)
        task = _crop_task(_tile_at(farm, x, y), crop, day, hour)
        if task is None:
            continue
        priority, action = task
        if action[0] == "PLANT":
            if seeds[crop] <= 0:
                continue
            seeds[crop] -= 1
        tasks.append({"priority": priority, "target": (x, y), "action": action})
    return tasks


def _assign_crop_actions(farm, private, day, hour, positions, unit_indices):
    actions = {}
    tasks = _crop_tasks(farm, private, day, hour)
    remaining = set(unit_indices)
    while tasks and remaining:
        choices = []
        for unit_index in remaining:
            ux, uy = positions[unit_index]
            for task_index, task in enumerate(tasks):
                tx, ty = task["target"]
                distance = abs(tx - ux) + abs(ty - uy)
                choices.append(
                    (
                        task["priority"], distance, ty, tx, unit_index, task_index,
                    )
                )
        _, _, _, _, unit_index, task_index = min(choices)
        task = tasks.pop(task_index)
        remaining.remove(unit_index)
        if positions[unit_index] == task["target"]:
            actions[unit_index] = task["action"]
        else:
            actions[unit_index] = _step_toward(positions[unit_index], task["target"])
    return actions


def _fertilizer_action(farm, private, day, position, unit_index):
    inventory = _unit_inventory(private, unit_index)
    carried = inventory.get("FERTILIZER", 0)
    targets = []
    for x, y, _ in CROP_SLOTS:
        tile = _tile_at(farm, x, y)
        if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
            continue
        if tile.get("fertilized_until_day", -1) >= day:
            continue
        crop = tile["crop"]
        age = day - tile["planted_day"]
        # A dose covers today and the following two daily refreshes.
        if crop == "TOMATO":
            worthwhile = 5 <= age <= 10
        elif crop == "STRAWBERRY":
            worthwhile = 7 <= age <= 15
        else:
            worthwhile = False
        if not worthwhile:
            continue
        price = farm.get("_market", {}).get("prices", {}).get(crop, 0)
        fertilizer_price = farm.get("_market", {}).get("prices", {}).get("FERTILIZER", 100)
        if 2 * price < fertilizer_price:
            continue
        distance = abs(x-position[0]) + abs(y-position[1])
        targets.append((distance, y, x))
    if carried and targets:
        _, y, x = min(targets)
        return ["FERTILIZE"] if position == (x,y) else _step_toward(position,(x,y))
    available = private.get("shed", {}).get("FERTILIZER",0)
    if not carried and available and targets:
        if _at_shed(position):
            return ["PICKUP", "FERTILIZER", min(4,available,len(targets))]
        return _step_toward(position, _nearest_shed(position))
    return None


def _assign_unit_actions(farm, private, day, hour):
    positions = [tuple(farm.get("farmer", (4, 4)))]
    positions.extend(tuple(pos) for pos in (farm.get("hands", []) or []))
    actions = [PASS for _ in positions]

    active_routes = (
        ANIMAL_ROUTES
        if len(farm.get("unlocked_quadrants", []) or []) > 1
        else ANIMAL_ROUTES[:BASE_ANIMAL_ROUTE_COUNT]
    )
    animal_role_count = min(len(active_routes), len(positions))
    crop_indices = []
    for unit_index in range(animal_role_count):
        route = active_routes[unit_index]
        action = _animal_route_action(farm,private,unit_index,route,day)
        if action is None:
            crop_indices.append(unit_index)
        else:
            actions[unit_index] = action

    for unit_index in range(animal_role_count, len(positions)):
        inventory = _unit_inventory(private, unit_index)
        carried = sum(max(0, _safe_int(inventory.get(item), 0)) for item in SELL_RULES)
        if day >= 29 and carried > 0:
            if _at_shed(positions[unit_index]):
                actions[unit_index] = ["DROP"]
            else:
                actions[unit_index] = _step_toward(positions[unit_index], _nearest_shed(positions[unit_index]))
        else:
            crop_indices.append(unit_index)

    if crop_indices:
        fertilizer_index = max(crop_indices)
        fertilizer_action = _fertilizer_action(
            farm, private, day, positions[fertilizer_index], fertilizer_index
        )
        if fertilizer_action is not None:
            actions[fertilizer_index] = fertilizer_action
            crop_indices.remove(fertilizer_index)

    for unit_index, action in _assign_crop_actions(farm, private, day, hour, positions, crop_indices).items():
        actions[unit_index] = action
    return actions


def _carried_totals(private):
    totals = {}
    for inventory in (private.get("inventories", []) or []):
        if not isinstance(inventory, dict):
            continue
        for item, quantity in inventory.items():
            totals[item] = totals.get(item, 0) + max(0, _safe_int(quantity))
    return totals


def _predicted_drop(private, unit_actions):
    totals = {}
    inventories = private.get("inventories", []) or []
    for index, action in enumerate(unit_actions):
        if not (isinstance(action, list) and action and action[0] == "DROP"):
            continue
        if index >= len(inventories) or not isinstance(inventories[index], dict):
            continue
        for item, quantity in inventories[index].items():
            totals[item] = totals.get(item, 0) + max(0, _safe_int(quantity))
    return totals


def _sell_orders(private, market, day, predicted_drop, limit):
    if limit <= 0:
        return []
    shed = private.get("shed", {}) or {}
    prices = (market or {}).get("prices", {}) or {}
    carried = _carried_totals(private)
    exposure = sum(max(0, _safe_int(v)) for v in shed.values()) + sum(carried.values())
    terminal = day >= LIQUIDATION_DAY
    forced = exposure >= 78
    orders = []

    for item in SELL_ORDER:
        dropping = _safe_int(predicted_drop.get(item), 0)
        held = _safe_int(shed.get(item), 0) + dropping
        if item == "FERTILIZER" and not terminal:
            held = max(0, held - FERTILIZER_RESERVE)
        if item == "WHEAT":
            carried_wheat = _safe_int(carried.get("WHEAT"), 0)
            non_dropping = max(0, carried_wheat - dropping)
            total_wheat = held + non_dropping
            reserve = (
                WHEAT_STOCKPILE_TARGET
                if WHEAT_STOCKPILE_DAY <= day < LIQUIDATION_DAY
                else (FEED_TARGET if day < 29 else 0)
            )
            held = min(held, max(0, total_wheat - reserve))
        if held <= 0:
            continue
        batch, floor = SELL_RULES[item]
        price = _safe_int(prices.get(item), 0)
        if terminal or forced or price >= floor:
            quantity = held if terminal else min(held, batch)
            orders.append(["SELL", item, quantity])
            if len(orders) >= limit:
                break
    return orders


def _planned_seed_needs(farm, private, day, hour):
    wanted = {crop: 0 for crop in CROPS}
    for x, y, crop in CROP_SLOTS:
        crop = _desired_crop(farm, x, y, crop, day)
        tile = _tile_at(farm, x, y)
        last_day = CROPS[crop]["last_plant_day"]
        before_cutoff = day < last_day or (
            day == last_day and hour < SEED_BUY_CUTOFF
        )
        if before_cutoff and (
            tile is None or (isinstance(tile, dict) and tile.get("kind") == "WEED")
        ):
            wanted[crop] += 1
    seeds = private.get("seeds", {}) or {}
    return {
        crop: max(0, wanted[crop] - _safe_int(seeds.get(crop), 0))
        for crop in CROPS
    }


def _procurement_orders(farm, private, day, hour, slots):
    if slots <= 0:
        return []
    orders = []
    cash = float(farm.get("money", 0))
    carried = _carried_totals(private)
    shed = private.get("shed", {}) or {}

    total_wheat = _safe_int(shed.get("WHEAT"), 0) + _safe_int(carried.get("WHEAT"), 0)
    wheat_price = max(
        1,
        _safe_int(
            ((farm.get("_market", {}) or {}).get("prices", {}) or {}).get("WHEAT"),
            30,
        ),
    )
    # Stage herd growth one animal per species per turn and preserve enough cash
    # to establish the feed buffer. This prevents expansion
    # capital from starving newly placed livestock before the first crop sale.
    feed_cash_floor = 80 + max(0, FEED_TARGET - total_wheat) * max(30, wheat_price + 4)
    if day <= 18:
        counts = _animal_counts(farm, private)
        active_routes = (
            ANIMAL_ROUTES
            if len(farm.get("unlocked_quadrants", []) or []) > 1
            else ANIMAL_ROUTES[:BASE_ANIMAL_ROUTE_COUNT]
        )
        desired = {
            animal: sum(
                1 for route in active_routes for _, _, kind in route
                if kind == animal
            )
            for animal in ANIMALS
        }
        for animal in ("GOOSE", "COW", "SHEEP"):
            missing = desired[animal] - counts[animal]
            cost = ANIMALS[animal]["cost"]
            if (
                missing > 0
                and len(orders) < slots
                and cash >= cost + feed_cash_floor
            ):
                orders.append(["BUY_ANIMAL", animal, 1])
                cash -= cost


    wheat_target = WHEAT_STOCKPILE_TARGET if day >= WHEAT_STOCKPILE_DAY else FEED_TARGET
    wheat_reorder = WHEAT_STOCKPILE_REORDER if day >= WHEAT_STOCKPILE_DAY else FEED_REORDER
    if day <= FINAL_CARE_DAY and total_wheat <= wheat_reorder and len(orders) < slots:
        price = max(1, _safe_int(((farm.get("_market", {}) or {}).get("prices", {}) or {}).get("WHEAT"), 30))
        quantity = wheat_target - total_wheat
        affordable = max(0, int((cash - 100) // max(30, price + 4)))
        quantity = min(quantity, affordable)
        if quantity > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", quantity])
            cash -= quantity * max(30, price + 4)

    if day >= 26 or len(orders) >= slots:
        return orders[:slots]
    needs = _planned_seed_needs(farm, private, day, hour)
    for crop in ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY"):
        if len(orders) >= slots:
            break
        quantity = needs[crop]
        if quantity <= 0:
            continue
        cost = CROPS[crop]["seed_cost"]
        affordable = max(0, int((cash - 80) // cost))
        quantity = min(quantity, affordable)
        if quantity <= 0:
            continue
        orders.append(["BUY_SEED", crop, quantity])
        cash -= quantity * cost
    return orders[:slots]


def _market_orders(farm, private, market, day, hour, predicted_drop):
    counts = _animal_counts(farm, private)
    core_ready = counts["COW"] >= 3 and counts["SHEEP"] >= 3
    buy_land = (
        EXPAND_LAND
        and core_ready
        and day <= LAND_BUY_CUTOFF
        and hour == 1
        and len(farm.get("unlocked_quadrants", []) or []) == 1
        and float(farm.get("money", 0)) >= FIRST_LAND_COST + LAND_CASH_BUFFER
    )

    # Reserve expansion cash before sizing purchases instead of depending on
    # partial fills after BUY_LAND consumes the first 1,000 coins.
    farm_for_buying = dict(farm)
    farm_for_buying["_market"] = market or {}
    if buy_land:
        farm_for_buying["money"] = (
            float(farm.get("money", 0)) - FIRST_LAND_COST
        )

    current_hands = len(farm.get("hands", []) or [])
    desired_hands = (
        EXPANDED_HANDS
        if len(farm.get("unlocked_quadrants", []) or []) > 1
        else BASE_HANDS
    )
    needed_hires = (
        max(0, desired_hands - current_hands)
        if hour == 0 and day <= 29 else 0
    )
    reserved_slots = needed_hires + int(buy_land)
    sale_limit = max(0, MAX_MARKET_ORDERS - reserved_slots)
    orders = _sell_orders(private, market, day, predicted_drop, sale_limit)

    for _ in range(needed_hires):
        orders.append(["HIRE"])

    free = MAX_MARKET_ORDERS - len(orders) - int(buy_land)
    if free > 0:
        orders.extend(
            _procurement_orders(farm_for_buying, private, day, hour, free)
        )

    if buy_land:
        orders.insert(0, ["BUY_LAND"])
    return orders[:MAX_MARKET_ORDERS]





def _price(crop, inventory):
    import math
    delta = inventory - 10000
    base = CROPS[crop]["base_price"]
    if crop == "WHEAT":
        return max(1,base-math.log1p(delta)*5/math.log(401)) if delta >= 0 else base+math.sqrt(-delta)
    if crop in ("CARROT","TOMATO"):
        throughput = 450 if crop == "CARROT" else 200
        down = .7 if crop == "CARROT" else .6
        up = 1 if crop == "CARROT" else .4
        u = abs(delta)/throughput
        return max(1,base*(1-down*math.sqrt(u))) if delta >= 0 else base*(1+up*(u+8*max(0,u-1)**2))
    if crop == "STRAWBERRY":
        return max(1,120-1.92*delta) if delta >= 0 else 120+8.4*math.sqrt(-delta)
    return max(1,250-.01*delta**2) if delta >= 0 else 250+50*math.log1p(-delta)/math.log(301)

def _crop_plan(farm, day):
    existing = farm.get("_crop_plan")
    if existing is not None:
        return existing
    schedules = {"WHEAT": ((4,4),), "CARROT": ((3,3),),
                 "MELON": ((10,6),), "TOMATO": ((8,2),(9,2),(10,2),(11,2)),
                 "STRAWBERRY": ((10,2),(12,2),(14,2),(16,2))}
    demand = {c:1.0 for c in CROPS}
    shop_items = {"BAKERY":("WHEAT",), "PIZZA_SHOP":("TOMATO","WHEAT"),
                  "BRUNCH_SPOT":("WHEAT","STRAWBERRY"),
                  "ICE_CREAM_SHOP":("STRAWBERRY","WHEAT"),
                  "SMOOTHIE_SHOP":("STRAWBERRY",), "PET_CAFE":("CARROT","CARROT"),
                  "FARMERS_MARKET":("WHEAT","CARROT","TOMATO","STRAWBERRY")}
    shops = farm.get("_town",{}).get("unlocked_shops",[])
    for shop in shops:
        for c in shop_items.get(shop,()):
            demand[c] += 6
    inventory = farm.get("_market",{}).get("inventory",{})
    # Project visible standing crops onto the same inventory curve used by
    # the game. Planned plots update the forecast before the next choice.
    output = {c:[0.0]*30 for c in CROPS}
    plan = {}
    for other in [farm,farm.get("_opponent",{})]:
        for y,row in enumerate(other.get("tiles",[])):
            for x,tile in enumerate(row):
                if not isinstance(tile,dict) or tile.get("kind") != "PLANT":
                    continue
                c = tile["crop"]
                planted = tile["planted_day"]
                if other is farm:
                    plan[(x,y)] = c
                held = tile.get("yield_units",0)
                if CROPS[c]["ongoing"]:
                    output[c][day] += held
                for age,units in schedules[c]:
                    at = planted+age
                    if at >= day and at < 30:
                        output[c][at] += units if other is farm else max(1,units//2) if CROPS[c]["ongoing"] else units
    points = sorted(CROP_SLOTS,key=lambda s:(abs(4-s[0])+abs(4-s[1]),s[1],s[0]))
    expected_shop = {"WHEAT":3.75,"CARROT":2.25,"TOMATO":1.5,"STRAWBERRY":3.,"MELON":0.}
    for x,y,planned in points:
        tile = _tile_at(farm,x,y)
        if (x,y) in plan or tile == "LOCKED":
            continue
        scores = []
        for c in CROPS:
            if day > CROPS[c]["last_plant_day"]:
                continue
            events = [(day+age,n) for age,n in schedules[c] if day+age <= 29]
            if not events:
                continue
            revenue = -CROPS[c]["seed_cost"]
            for at,units in events:
                horizon = at-day
                new_shops = min(8-len(shops),horizon/6)
                anticipated_demand = horizon * (demand[c] + new_shops*expected_shop[c])
                supply = sum(output[c][day:at+1])
                projected = inventory.get(c,10000)-anticipated_demand+supply+units/2
                revenue += units*_price(c,projected)
            duration = events[-1][0]-day+1
            score = revenue/duration
            # Favor cheap feed crops in the opening when their expected
            # return is competitive with the other crop choices.
            if day < 4 and c == "WHEAT":
                score *= 1.7
            scores.append((score,c))
        crop = max(scores)[1] if scores else "WHEAT"
        plan[(x,y)] = crop
        for age,units in schedules[crop]:
            at = day+age
            if at < 30:
                output[crop][at] += units
    farm["_crop_plan"] = plan
    return plan

CROP_SLOTS = CROP_SLOTS + tuple((x,y,"WHEAT") for y in range(5) for x in range(5,10)
    if (x,y) not in _ANIMAL_POINTS and not any(s[:2] == (x,y) for s in CROP_SLOTS))


def agent(obs):
    """Required Kaggle entrypoint."""
    try:
        farms = obs.get("farms", []) or []
        player = _safe_int(obs.get("player"), 0)
        private = obs.get("private", {}) or {}
        if player < 0 or player >= len(farms):
            return {"farmer": PASS, "hands": [], "market": []}
        farm = farms[player]
        day = _safe_int(obs.get("day"), 0)
        hour = _safe_int(obs.get("hour"), 0)
        market = obs.get("market", {}) or {}
        farm["_market"] = market
        farm["_town"] = obs.get("town", {}) or {}
        farm["_opponent"] = farms[1-player]

        unit_actions = _assign_unit_actions(farm, private, day, hour)
        predicted = _predicted_drop(private, unit_actions)
        market_orders = _market_orders(farm, private, market, day, hour, predicted)
        return {
            "farmer": unit_actions[0] if unit_actions else PASS,
            "hands": unit_actions[1:],
            "market": market_orders,
        }
    except Exception:
        hands = []
        try:
            farms = obs.get("farms", []) or []
            player = _safe_int(obs.get("player"), 0)
            if 0 <= player < len(farms):
                hands = [PASS for _ in (farms[player].get("hands", []) or [])]
        except Exception:
            hands = []
        return {"farmer": PASS, "hands": hands, "market": []}
