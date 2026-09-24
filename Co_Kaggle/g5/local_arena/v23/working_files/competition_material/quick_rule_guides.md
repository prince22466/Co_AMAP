one order = one action + one product + quantity.
For example, selling 20 milk and buying 5 wheat are two orders. HIRE and BUY_LAND are exceptions: each is one order without a product or quantity.



up to 10 market orders per player, per turn.
An order can cover multiple units: SELL MILK 20 counts as one order, not 20. Buying seeds or animals also supports quantities.
Hiring and buying land each use an order slot. These market orders are separate from workers’ actions.



In the same turn:
- Your farmer can move.
- One hired worker can pick up an animal.
- Another can water a crop.
- Another can harvest.
- You can also submit up to 10 market orders, such as buying and selling.
But the same worker cannot pick up, move, and place in one turn. Those require separate turns; each movement step is also one action.



Crop production
Times below start from the planting day, assuming the plant survives.
Product	Seed cost	First harvest age	Production cycle	Yield
Wheat	10	2 days ≈48 turns	One harvest; growth window ends at age 4 days	Up to 4 without fertilizer, 6 with fertilizer
Carrot	20	2 days ≈48 turns	One harvest; growth window ends at age 3 days	Up to 3 without fertilizer, 4 with fertilizer
Melon	80	10 days ≈240 turns	One harvest; growth window ends at age 12 days	Up to 6
Tomato	50	8 days ≈192 turns	Four production events: ages 8, 9, 10, 11 days	1 per event, or 2 when watered and fertilized
Strawberry	100	10 days ≈240 turns	Four production events: ages 10, 12, 14, 16 days	1 per event, or 2 when watered and fertilized


These are calendar-day ages, not exact elapsed turns. For example, wheat planted at hour 20 reaches age two after 28 turns, not 48.
Important details:
- Seeds enter shared seed inventory; planting requires a separate worker action.
- Water on the planting day. Otherwise the plant can become weeds at the first day boundary. Subsequently, two consecutive unwatered days kill it.
- Early harvesting ends a wheat/carrot/melon crop, potentially sacrificing later yield.
- Tomato/strawberry store at most 4 unharvested units. Delayed harvesting can waste production.
- One fertilizer application consumes 1 fertilizer and lasts three calendar days.




Animal production
Times start from placement on a suitable structure, not purchase.
Product	Animal purchase	First production age	Repeat interval	Production with daily feeding and care*
Eggs	Goose: 300	4 days ≈96 turns	1 day =24 turns	First batch up to 4, then 2/event
Milk	Cow: 400	8 days ≈192 turns	2 days =48 turns	First batch up to 6, then 3/event
Wool	Sheep: 500	6 days ≈144 turns	3 days =72 turns	First batch up to 6, then 4/event
Fertilizer	By-product of any animal	Next day boundary after placement	Available daily	1 per collection; missed days do not accumulate


*Assumes care starts on placement day and harvesting prevents storage overflow. Without care bonuses, animal production is 1 unit/event.
Animals require:
- Purchase → shed → worker pickup → transport → PLACE.
- A matching pasture or coop. Building costs a worker action, but no cash/materials.
- 1 wheat per daily feeding, plus a separate CARE action for bonuses.
- Two consecutive unfed days cause escape.
- Fertilizer requires a separate collection action.




From harvest to sale
For every harvested product:
HARVEST / COLLECT
    → worker inventory
    → travel to shed and DROP
    → eligible for SELL
Each move and operation consumes a worker action. Worker actions run before market orders, so a successful DROP can be sold in the same turn.
Alternatively, carried inventory automatically enters the shed at day-end, subject to capacity; overflow is discarded. Holding rules can then delay selling further.
Therefore:
Buying-to-sale time = setup/waiting + growth + harvest scheduling + delivery + selling delay.




End-to-end costs
There is no fixed engine charge for watering, caring, harvesting, transporting, or selling. Their cost is worker time and any associated hiring expense.
Product	Direct production-cost calculation
Wheat, carrot, melon, tomato, strawberry	Seed cost + fertilizer consumed
Eggs	Allocated goose purchase cost + wheat feed
Milk	Allocated cow purchase cost + wheat feed
Wool	Allocated sheep purchase cost + wheat feed
Fertilizer	A share of animal purchase/feed costs; it is a joint product


Then add allocated worker hiring and land costs to obtain a full cost.
For example, charging the entire cow purchase to its first batch:
First milk batch cost = 400 + cost of approximately 8 daily wheat feeds + allocated labor/land.
Later batches do not require another cow purchase. Animal fertilizer revenue also offsets the combined operation’s cost.
Two cautions matter for our analysis:
- Homegrown wheat/fertilizer is not economically free: consuming it gives up potential sale revenue.
- The planner’s LABOR_COST = 20 is a forecasting assumption, not an engine fee per care action.





Each turn, you can observe the opponent’s public farm state:
Visible	Details
Money	Current cash
Workers	Farmer and hired workers’ positions, worker count
Hiring	Number hired that day
Land	Unlocked quadrants
Crops	Type, planting day, watering/fertilizer status, yield on the tile
Animals	Type, placement day, feeding/care status, available yield
Farm layout	Structures, weeds, and other tile contents


You cannot directly see their:
- Shed contents.
- Seed inventory.
- Goods carried by workers.
- Submitted orders or intended next actions.


for survival, feeding every other day is enough. The animal escapes only after two consecutive unfed days.
But there’s a production tradeoff:
- CARE only builds a bonus on days the animal is both fed and cared for.
- On an unfed production day, it produces only the base 1 unit, and accumulated CARE bonuses are cleared.
So alternating feeding days saves wheat and worker actions, but can reduce milk, wool, or egg output. Daily feeding supports higher production; it isn’t strictly required for survival.

hired workers leave at the end of each day; your farmer stays. You choose how many to hire again the next day.
A few details from the local game engine:
- You can hire during the day too, not just at the beginning.
- Each HIRE order adds one worker, provided you can afford it and stay within the market-order limit.
- Successive hires cost 1, 1, 2, 3, 5, 8, … times the configured hiring-cost multiplier. This resets daily.
- New workers can act starting the next turn, because hiring happens after worker actions.
So hiring early gives each worker more turns to work.



A worker can collect fertilizer from animals, carry it to the shed, and deposit it. You can then sell it to the shared market.
You can also keep fertilizer to use on crops. It counts toward the shed’s 100-unit capacity.





You can observe more than prices:
Information	Visible?
Current product prices in the shared market	Yes
Quantity of each product in the shared market	Yes
Which shops are open, including duplicates	Yes
Products and quantities those shops consume	Known from the game rules
Which random shop will open next	No
Opponent’s crops and animals	Yes—useful for estimating future supply
Opponent’s private stored/carried goods	No


One naming distinction: “Farmers Market” is one shop type. The shared market is where you trade. Shops don’t have separate trading prices for you.



you cannot sell directly to shops.
You sell to the shared market and get paid immediately. Shops automatically consume goods from that market later.



Town Center demand
In addition to unlocked shops, the Town Center consumes products from the shared market throughout the entire game.
- It consumes 1 unit of every non-fertilizer product every 24 turns by default — once per in-game day.
- This demand exists from the start of the game and does not depend on which shops have unlocked.
- Fertilizer is not consumed by the Town Center.
- Town Center consumption removes goods from shared market inventory, which can affect market prices.
- The Town Center does not buy directly from players. Players sell to the shared market; the Town Center later consumes goods from that market.
Therefore, even when no shops are unlocked, there is still baseline demand for wheat, carrot, tomato, strawberry, melon, eggs, milk, and wool.



The shops buy from the shared market 6 times per day, once every 4 turns. That schedule doesn’t restrict when you sell.


There are 8 shop types. They buy/consume products from the shared market; they don’t sell goods to you directly.


Each shop consumes products once every 4 turns (4 in-game hours):
Shop	Amount per buying cycle
Bakery	1 wheat + 1 egg
Pizza shop	1 milk + 1 tomato + 1 wheat
Brunch spot	1 egg + 1 wheat + 1 strawberry
Yarn store	2 wool
Ice cream shop	1 strawberry + 1 milk + 1 wheat
Pet café	2 carrots
Smoothie shop	1 strawberry + 1 milk
Farmers market	1 wheat + 1 carrot + 1 tomato + 1 strawberry





new shops open automatically as the game progresses, up to a limit:
- Starts with 0 shops.
- One random shop opens every 3 days.
- Maximum 8 shops total.
- The same shop type can appear multiple times, such as two bakeries.
Shops consume products from the shared market, increasing demand. For example, more bakeries means more demand for wheat and eggs, which can help their prices rise.



Animal purchase prices are also fixed throughout the game:
Animal	Price
Cow	400
Sheep	500
Goose	300




Selling requires goods in the shed. Workers act before market orders are processed, so a worker can deposit goods and you can sell them in the same turn.



Anything that exceeds the shed’s 100-unit capacity is discarded permanently.



Each worker can perform one action per turn.
Activity	Time required
Travel	1 turn per tile, horizontally or vertically
Plant	1 turn for one crop tile
Water	1 turn for one crop tile
Feed	1 turn for one animal, carrying wheat
Harvest	1 turn to collect the available yield from one tile
Deliver	Travel to the shed + 1 turn to deposit goods



The hiring price increases with each hand you hire that day, rather than fluctuating with the market.
Costs follow this sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89…
For example, hiring 6 hands costs 20 total. The sequence resets the next day, when you need to hire hands again.