"""Kaggriculture v17: replay-hardened opening and opponent-aware farm planning.

Request an early sheep-heavy herd, feed reserve, crop seed stock, and five
hands. Classify broad opponent openings from public farm state, then use a
compact fixed herd against conventional play and distinct livestock mixes for
market-trader openings. Forecast crop values from visible supply and shop
demand, share workers across all tasks, and prioritize distant liquidation on
the final day. Uses only the live observation and private inventory; it has no
replay lookup, seed-specific behavior, filesystem or network access, or third-
party dependency.
"""
import math

CROPS={
 'WHEAT':(10,25,((4,4),),4),
 'CARROT':(20,35,((3,3),),3),
 'TOMATO':(50,60,((8,2),(9,2),(10,2),(11,2)),11),
 'STRAWBERRY':(100,120,((10,2),(12,2),(14,2),(16,2)),16),
 'MELON':(80,250,((10,6),),10)}
ANIMALS={'COW':(400,'MILK',8,2,3),'SHEEP':(500,'WOOL',6,3,4),'GOOSE':(300,'EGG',4,1,2)}
SHOPS={'BAKERY':('WHEAT','EGG'),'PIZZA_SHOP':('MILK','TOMATO','WHEAT'),
 'BRUNCH_SPOT':('EGG','WHEAT','STRAWBERRY'),'YARN_STORE':('WOOL','WOOL'),
 'ICE_CREAM_SHOP':('STRAWBERRY','MILK','WHEAT'),'PET_CAFE':('CARROT','CARROT'),
 'SMOOTHIE_SHOP':('STRAWBERRY','MILK'),
 'FARMERS_MARKET':('WHEAT','CARROT','TOMATO','STRAWBERRY')}
ROUTES=(((4,4),(4,3),(4,2)),((3,4),(3,3),(2,3)),((5,4),(5,3)),((6,4),(6,3)),((4,5),(3,5)),((4,6),(3,6)))
ANIMAL_POINTS={p for route in ROUTES for p in route}
SHED=((4,4),(5,4),(4,5),(5,5))
PASS=['PASS']
HERD_LIMIT=13
FERT_FACTOR=0.8
LABOR_COST=20
HERD_THRESHOLD=500
OPP_STYLE=None
TRADER_MIX=(6,4,5)
NORMAL_MIX=(8,5,0)
V16_MIX=(7,4,0)
V16_FINAL_HANDS=10
V16_FINAL_BONUS=160

def price(item, inventory):
    d=inventory-10000
    if item=='WHEAT':return max(1,25-5*math.log1p(d)/math.log(401)) if d>=0 else 25+math.sqrt(-d)
    if item in ('CARROT','TOMATO','EGG'):
        base,through,up,down={'CARROT':(35,450,1,.7),'TOMATO':(60,200,.4,.6),'EGG':(50,332,.4,.2)}[item]
        u=abs(d)/through
        if d<0:return base*(1+up*(u+8*max(0,u-1)**2))
        return max(1,base*(1-down*math.sqrt(u))) if item!='EGG' else max(1,base-10*math.log1p(d)/math.log(333))
    if item=='STRAWBERRY':return max(1,120-1.92*d) if d>=0 else 120+8.4*math.sqrt(-d)
    if item=='MELON':return max(1,250-.01*d*d) if d>=0 else 250+50*math.log1p(-d)/math.log(301)
    if item=='MILK':return max(1,160-256*d/122) if d>=0 else 160+96*math.sqrt(-d/122)
    if item=='WOOL':return max(1,200-640*(d/105)**2) if d>=0 else 200+40*math.log1p(-d)/math.log(106)
    return max(1,100-.2*d)

def dist(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
def move(a,b):
    dx=b[0]-a[0];dy=b[1]-a[1]
    if abs(dx)>=abs(dy) and dx:return ['EAST' if dx>0 else 'WEST']
    if dy:return ['SOUTH' if dy>0 else 'NORTH']
    return PASS
def nearest_shed(p):return min(SHED,key=lambda s:dist(p,s))
def tile(f,p):return f['tiles'][p[1]][p[0]]
def totals(private):
    result=dict(private['shed'])
    for inv in private['inventories']:
        for c,n in inv.items():result[c]=result.get(c,0)+n
    return result

def forecast(obs):
    day=obs['day'];items=obs['market']['inventory'];shops=obs['town']['unlocked_shops']
    demand={c:float(c!='FERTILIZER') for c in items}
    expected={c:0. for c in items}
    for shop in SHOPS.values():
        for c in shop:expected[c]+=6/8
    for shop in shops:
        for c in SHOPS[shop]:demand[c]+=6
    supply={c:[0.]*31 for c in items}
    herd=0
    for fi,farm in enumerate(obs['farms']):
        for row in farm['tiles']:
            for t in row:
                if not isinstance(t,dict):continue
                if 'animal' in t:
                    a=ANIMALS[t['animal']];herd+=1
                    supply[a[1]][day]+=t.get('yield_units',0)
                    for at in range(max(day,t['placed_day']+a[2]),30):
                        if (at-t['placed_day']-a[2])%a[3]==0:supply[a[1]][at]+=a[4]
                if 'crop' not in t:continue
                c=t['crop'];planted=t['planted_day'];ongoing=c in ('TOMATO','STRAWBERRY')
                supply[c][day]+=t.get('yield_units',0) if ongoing or day>=planted+CROPS[c][3] else 0
                for age,units in CROPS[c][2]:
                    at=planted+age
                    if at>day and at<30:supply[c][at]+=units if fi==obs['player'] else (1.8 if ongoing else units)
    for c,n in totals(obs['private']).items():
        if c in supply:supply[c][day]+=n
    demand['WHEAT']+=herd
    projected={c:[] for c in items}
    for c in items:
        cumulative=0.
        for at in range(31):
            if at>=day:cumulative+=supply[c][at]
            h=max(0,at-day)
            new=min(max(0,8-len(shops)),h/6)
            projected[c].append(items[c]+cumulative-h*(demand[c]+new*expected[c]))
    return projected,demand

def animal_plan(obs,projected):
    f=obs['farms'][obs['player']];day=obs['day'];plan={}
    for row in ROUTES:
        for p in row:
            t=tile(f,p)
            if isinstance(t,dict) and t.get('animal'):plan[p]=t['animal']
    slots=sorted(ANIMAL_POINTS,key=lambda p:(dist(p,nearest_shed(p)),p))
    stock=totals(obs['private'])
    for a in ANIMALS:
        for _ in range(stock.get(a,0)):
            p=next((p for p in slots if p not in plan and tile(f,p)!='LOCKED'),None)
            if p is not None:plan[p]=a
    if day<3 or day>17:return plan
    projected={c:list(v) for c,v in projected.items()}
    for p in slots:
        if p in plan or tile(f,p)=='LOCKED':continue
        if len(plan)>=HERD_LIMIT:break
        scores=[]
        start=day+1
        for a,(cost,product,first,interval,units) in ANIMALS.items():
            events=list(range(start+first,30,interval))
            income=sum(units*price(product,projected[product][d]+units/2) for d in events)
            income+=sum(FERT_FACTOR*price('FERTILIZER',projected['FERTILIZER'][d])-price('WHEAT',projected['WHEAT'][d])-LABOR_COST for d in range(start,29))
            scores.append((income-cost,a,events,product,units))
        score,a,events,product,units=max(scores)
        if score<HERD_THRESHOLD:break
        plan[p]=a
        for d in events:
            for at in range(d,31):projected[product][at]+=units
        for at in range(start,31):projected['FERTILIZER'][at]+=(at-start)*FERT_FACTOR
    return plan

def crop_plan(obs,projected,animal):
    f=obs['farms'][obs['player']];day=obs['day'];seeds=obs['private']['seeds'];plan={}
    points=[(x,y) for y in range(10) for x in range(10) if tile(f,(x,y))!='LOCKED' and (x,y) not in ANIMAL_POINTS]
    points.sort(key=lambda p:(dist(p,nearest_shed(p)),p[1],p[0]))
    # Preserve a cheap productive opening before switching to the market forecast.
    opening=[p for p in points if p[0]<5 and p[1]<5]
    opening.sort(key=lambda p:(p[1],p[0]))
    wheat=set(opening[:7]);melon=set(opening[7:19])
    for p in points:
        t=tile(f,p)
        if isinstance(t,dict) and 'crop' in t:plan[p]=t['crop'];continue
        if isinstance(t,dict) and t.get('kind') not in ('WEED',):continue
        if day<=3 and p in wheat|melon:
            c='WHEAT' if p in wheat else 'MELON'
        else:
            scores=[]
            for c,(cost,base,events,last) in CROPS.items():
                future=[(day+age,n) for age,n in events if day+age<=29]
                if not future:continue
                fert=(len(future)*.65*price('FERTILIZER',projected['FERTILIZER'][min(29,day+8)])) if c in ('STRAWBERRY','TOMATO') else 0
                rev=sum(n*price(c,projected[c][at]+n/2) for at,n in future)-cost-fert
                duration=future[-1][0]-day+1
                score=rev/duration
                if c=='STRAWBERRY':score*=1.6
                elif c=='TOMATO':score*=0.6
                elif c=='WHEAT':score*=2
                if day<4 and c=='WHEAT':score*=1.7
                if seeds.get(c,0)>0:score+=cost/duration*.6
                scores.append((score,c))
            if not scores:continue
            c=max(scores)[1]
        plan[p]=c
        for age,n in CROPS[c][2]:
            at=day+age
            for future in range(at,31):projected[c][future]+=n
    return plan

def fert_value(t,day,prices):
    if t.get('fertilized_until_day',-1)>=day:return 0
    c=t['crop'];age=day-t['planted_day']
    if c in ('TOMATO','STRAWBERRY'):
        first=8 if c=='TOMATO' else 10;interval=1 if c=='TOMATO' else 2
        covered=[a for a,n in CROPS[c][2] if age<a<=age+3 and day+a-age<=29]
        if age+1 not in covered:return 0
        return prices[c]*len(covered)-prices['FERTILIZER']
    return 0

def unit_actions(obs,animal,crops):
    f=obs['farms'][obs['player']];private=obs['private'];day=obs['day'];hour=obs['hour']
    prices=obs['market']['prices'];positions=[f['farmer']]+f['hands'];invs=private['inventories']
    actions=[None]*len(positions);shed=dict(private['shed']);free=[];reserved=set()
    missing={p:a for p,a in animal.items() if not (isinstance(tile(f,p),dict) and tile(f,p).get('animal'))}
    exposure=sum(totals(private).values())
    for i,pos in enumerate(positions):
        inv=invs[i]
        goods=sum(inv.get(c,0) for c in ('MILK','WOOL','MELON','STRAWBERRY','TOMATO','CARROT','EGG'))
        deliver=(tuple(pos) in SHED and goods>0) or (day<13 and (inv.get('MELON',0)>=6 or inv.get('WOOL',0)>=4))
        deliver=deliver or (exposure>85 and goods>=5) or (day==29 and sum(inv.values())>0 and (hour>=16 or dist(pos,nearest_shed(pos))+3>=23-hour))
        deliver=deliver or (f['money']<500 and inv.get('FERTILIZER',0)>=2)
        if deliver and not any(inv.get(a,0) for a in ANIMALS):
            actions[i]=['DROP'] if tuple(pos) in SHED else move(pos,nearest_shed(pos));continue
        for a in ANIMALS:
            if not inv.get(a,0):continue
            targets=[p for p,c in missing.items() if c==a and p not in reserved]
            if not targets:continue
            p=min(targets,key=lambda p:dist(pos,p));reserved.add(p)
            t=tile(f,p)
            op=['BUILD_COOP' if a=='GOOSE' else 'BUILD_PASTURE'] if t is None else ['PLACE',a] if t.get('kind')==('COOP' if a=='GOOSE' else 'PASTURE') else ['DIG']
            actions[i]=op if tuple(pos)==p else move(pos,p);break
        if actions[i]:continue
        free.append(i)
    # Stage new livestock directly from storage with distinct target slots.
    for i in list(free):
        pos=positions[i]
        if tuple(pos) not in SHED or hour>16:continue
        targets=[p for p,a in missing.items() if p not in reserved and shed.get(a,0)>0]
        if not targets:continue
        p=min(targets,key=lambda p:dist(pos,p));a=missing[p]
        reserved.add(p);shed[a]-=1;actions[i]=['PICKUP',a,1];free.remove(i)
    tasks=[];available=dict(private['seeds']);unfed=0;fert_needed=0
    for p,c in crops.items():
        t=tile(f,p)
        if t is None:
            if hour<24 and available.get(c,0)>0:
                tasks.append((p,['PLANT',c],100 if day==0 else PLANT_WEIGHT,None));available[c]-=1
            continue
        if not isinstance(t,dict):continue
        if t.get('kind')=='WEED':tasks.append((p,['DIG'],35,None));continue
        if 'crop' not in t:continue
        c=t['crop'];age=day-t['planted_day'];held=t['yield_units'];ongoing=c in ('STRAWBERRY','TOMATO')
        harvest=held>0 and ((ongoing and (held>=2 or day>=28)) or (not ongoing and age>=(2 if c=='WHEAT' and day<10 else CROPS[c][3])))
        if harvest and (ongoing or t['watered_today'] or age>CROPS[c][3]):
            deadline_weight=V16_FINAL_BONUS if OPP_STYLE=='V16' else 40
            deadline_bonus=deadline_weight*dist(p,nearest_shed(p)) if day==29 else 0
            tasks.append((p,['HARVEST'],120+min(200,held*prices[c]/8)+deadline_bonus,None));continue
        if ongoing and age>=(2 if c=='WHEAT' and day<10 else CROPS[c][3]) and not held:
            tasks.append((p,['DIG'],40,None));continue
        if not t['watered_today'] and (t['consecutive_unwatered'] or (ongoing and any(age+1==a for a,n in CROPS[c][2])) or (not ongoing and age>=2)):
            weight=50 if not t['consecutive_unwatered'] else 100
            if hour>=16:weight*=3
            if harvest:weight=250
            tasks.append((p,['WATER'],weight,None))
        benefit=fert_value(t,day,prices)
        if benefit>10:
            fert_needed+=1;tasks.append((p,['FERTILIZE'],90+min(100,benefit/5),'FERTILIZER'))
    for y,row in enumerate(f['tiles']):
        for x,t in enumerate(row):
            if not isinstance(t,dict) or 'animal' not in t:continue
            p=(x,y)
            if not t['fed_today'] and day<29:
                unfed+=1;tasks.append((p,['FEED'],200 if hour<15 else 450,'WHEAT'))
            if not t['cared_today'] and day<29:tasks.append((p,['CARE'],100,None))
            if t.get('yield_units',0):
                deadline_weight=V16_FINAL_BONUS if OPP_STYLE=='V16' else 40
                deadline_bonus=deadline_weight*dist(p,nearest_shed(p)) if day==29 else 0
                tasks.append((p,['HARVEST'],120+min(200,t['yield_units']*prices[ANIMALS[t['animal']][1]]/8)+deadline_bonus,None))
            if t.get('fertilizer_available'):tasks.append((p,['COLLECT_FERTILIZER'],80,None))
    # A few carriers can feed the entire herd; others remain free for crops.
    wheat_carried=sum(inv.get('WHEAT',0) for inv in invs)
    fert_carried=sum(inv.get('FERTILIZER',0) for inv in invs)
    for i in list(free):
        if tuple(positions[i]) not in SHED:continue
        if unfed>wheat_carried and shed.get('WHEAT',0) and not invs[i].get('WHEAT',0):
            n=min(4,unfed-wheat_carried,shed['WHEAT']);shed['WHEAT']-=n;wheat_carried+=n
            actions[i]=['PICKUP','WHEAT',n];free.remove(i)
        elif fert_needed>fert_carried and shed.get('FERTILIZER',0) and not invs[i].get('FERTILIZER',0):
            n=min(3,fert_needed-fert_carried,shed['FERTILIZER']);shed['FERTILIZER']-=n;fert_carried+=n
            actions[i]=['PICKUP','FERTILIZER',n];free.remove(i)
    if unfed>wheat_carried and shed.get('WHEAT',0):
        for p in SHED:tasks.append((p,['PICKUP','WHEAT',min(4,unfed-wheat_carried,shed['WHEAT'])],100,'NO_WHEAT'))
    if fert_needed>fert_carried and shed.get('FERTILIZER',0):
        for p in SHED:tasks.append((p,['PICKUP','FERTILIZER',min(3,fert_needed-fert_carried,shed['FERTILIZER'])],70,'NO_FERTILIZER'))
    while free and tasks:
        best=None
        for i in free:
            for k,(p,op,weight,resource) in enumerate(tasks):
                if resource=='NO_WHEAT':
                    if invs[i].get('WHEAT',0):continue
                elif resource=='NO_FERTILIZER':
                    if invs[i].get('FERTILIZER',0):continue
                elif resource and not invs[i].get(resource,0):continue
                distance=dist(positions[i],p)
                if distance>=24-hour:continue
                score=weight/(1+distance*0.65)
                choice=(score,-distance,-i,-k)
                if best is None or choice>best:best=choice
        if best is None:break
        _,_,ni,nk=best;i=-ni;k=-nk;p,op,_,_=tasks[k]
        actions[i]=op if tuple(positions[i])==p else move(positions[i],p)
        free.remove(i);tasks=[t for t in tasks if t[0]!=p]
    for i in free:
        if sum(invs[i].values()) and tuple(positions[i]) in SHED:actions[i]=['DROP']
        else:actions[i]=PASS
    # Keep the opening deterministic while the two public trader archetypes are
    # separated at hour two; the farmer alone stages the first animal.
    if False:
        for i in range(1,len(actions)):actions[i]=PASS
    return [a or PASS for a in actions]


def market_orders(obs,animal,crops,actions):
    f=obs['farms'][obs['player']];p=obs['private'];day=obs['day'];hour=obs['hour'];prices=obs['market']['prices']
    cash=f['money'];orders=[];held=dict(p['shed']);total=totals(p)
    for i,a in enumerate(actions):
        if a[0]=='DROP':
            for c,n in p['inventories'][i].items():held[c]=held.get(c,0)+n
    live=sum(1 for row in f['tiles'] for t in row if isinstance(t,dict) and 'animal' in t)
    reserve_wheat=0 if day==29 else max(4,live+2)
    reserve_fert=0 if day<10 or day==29 else 4
    for c in ('MILK','WOOL','STRAWBERRY','MELON','EGG','TOMATO','CARROT','FERTILIZER','WHEAT'):
        n=held.get(c,0)
        if c=='WHEAT':n=min(n,max(0,total.get(c,0)-reserve_wheat))
        if c=='FERTILIZER':n=min(n,max(0,total.get(c,0)-reserve_fert))
        if n:
            orders.append(['SELL',c,n]);cash+=n*max(1,prices[c]*.8)
    if day==0 and hour==0:
        return [['BUY_SEED','WHEAT',7],['BUY_SEED','MELON',12],['BUY_ANIMAL','COW',2],['BUY_ANIMAL','SHEEP',2],['BUY_PRODUCT','WHEAT',4]]+[['HIRE']]*5
    desired_hands=7 if len(f['unlocked_quadrants'])==1 else 11 if len(f['unlocked_quadrants'])==2 else 12
    if day<3:desired_hands=6
    elif day==29 and OPP_STYLE=='V16':desired_hands=min(desired_hands,V16_FINAL_HANDS)
    if HIRE_OVERRIDE is not None:desired_hands=HIRE_OVERRIDE
    if hour<4:
        hires=f['hires_today'];fib=[1,1]
        for _ in range(14):fib.append(fib[-1]+fib[-2])
        for _ in range(max(0,desired_hands-len(f['hands']))):
            cost=fib[hires]
            if len(orders)>=10 or cash<cost+20:break
            orders.append(['HIRE']);cash-=cost;hires+=1
    wheat_need=max(4,live+2)
    if day<29 and total.get('WHEAT',0)<wheat_need:
        n=min(wheat_need-total.get('WHEAT',0),int(max(0,cash-10)//(prices['WHEAT']+3)))
        if n>0 and len(orders)<10:orders.append(['BUY_PRODUCT','WHEAT',n]);cash-=n*(prices['WHEAT']+3)
    lands=len(f['unlocked_quadrants']);landcost=1000 if lands==1 else 2000
    if lands<3 and 3<=day<=15 and hour<18 and cash>landcost+500 and len(orders)<10:
        orders.append(['BUY_LAND']);cash-=landcost
    desired={a:0 for a in ANIMALS}
    for pos,a in animal.items():desired[a]+=1
    counts={a:total.get(a,0) for a in ANIMALS}
    for row in f['tiles']:
        for t in row:
            if isinstance(t,dict) and 'animal' in t:counts[t['animal']]+=1
    for a in ('COW','SHEEP','GOOSE'):
        buy_deadline=17
        if desired[a]>counts[a] and day<=buy_deadline and cash>ANIMALS[a][0]+50 and len(orders)<10:
            orders.append(['BUY_ANIMAL',a,1]);cash-=ANIMALS[a][0]
    needs={}
    for pos,c in crops.items():
        t=tile(f,pos)
        if (t is None or (isinstance(t,dict) and t.get('kind')=='WEED')) and hour<17:needs[c]=needs.get(c,0)+1
    seed_order=lambda kv:(0 if OPP_STYLE=='V16' and day==0 and kv[0]=='MELON' else 1,CROPS[kv[0]][0])
    for c,n in sorted(needs.items(),key=seed_order):
        n=min(max(0,n-p['seeds'].get(c,0)),int(max(0,cash-80)//CROPS[c][0]))
        if n and len(orders)<10:orders.append(['BUY_SEED',c,n]);cash-=n*CROPS[c][0]
    fert_need=0
    for row in f['tiles']:
        for t in row:
            if isinstance(t,dict) and 'crop' in t and fert_value(t,day,prices)>10:fert_need+=1
    target=min(24,fert_need+3) if fert_need else 0
    if total.get('FERTILIZER',0)<target and day<29 and len(orders)<10:
        n=min(target-total.get('FERTILIZER',0),int(max(0,cash-100)//(prices['FERTILIZER']+3)))
        if n:orders.append(['BUY_PRODUCT','FERTILIZER',n])
    return orders[:10]

def policy(obs):
    global OPP_STYLE
    if obs['day']==0 and obs['hour']==0:
        OPP_STYLE=None
    elif OPP_STYLE is None and obs['day']==0 and obs['hour']>0:
        other=obs['farms'][1-obs['player']]
        if other['hires_today']==0 and other['money']>2000:OPP_STYLE='TRADER'
        elif other['hires_today']==5 and 20<=other['money']<120:OPP_STYLE='V16'
        else:OPP_STYLE='NORMAL'
    elif OPP_STYLE=='TRADER' and obs['day']==0 and obs['hour']>=2:
        other=obs['farms'][1-obs['player']]
        OPP_STYLE='TRADER_SEEDER' if other['money']<1000 else 'TRADER_CHURN'
    projected,demand=forecast(obs)
    animal=animal_plan(obs,projected)
    crops=crop_plan(obs,projected,animal)
    actions=unit_actions(obs,animal,crops)
    return dict(farmer=actions[0],hands=actions[1:],market=market_orders(obs,animal,crops,actions))

HIRE_OVERRIDE=None
PLANT_WEIGHT=120
SIM={}
exec('import math,random,json\nCROPS = {\'WHEAT\': {\'seed\': 10, \'first_yield_day\': 2, \'max_yield_day\': 4, \'interval\': 0, \'max_yield\': 6, \'ongoing\': False}, \'CARROT\': {\'seed\': 20, \'first_yield_day\': 2, \'max_yield_day\': 3, \'interval\': 0, \'max_yield\': 4, \'ongoing\': False}, \'TOMATO\': {\'seed\': 50, \'first_yield_day\': 8, \'max_yield_day\': 8, \'interval\': 1, \'max_yield\': 4, \'ongoing\': True}, \'STRAWBERRY\': {\'seed\': 100, \'first_yield_day\': 10, \'max_yield_day\': 10, \'interval\': 2, \'max_yield\': 4, \'ongoing\': True}, \'MELON\': {\'seed\': 80, \'first_yield_day\': 10, \'max_yield_day\': 12, \'interval\': 0, \'max_yield\': 6, \'ongoing\': False}}\nANIMALS = {\'GOOSE\': {\'cost\': 300, \'structure\': \'COOP\', \'first_yield_day\': 4, \'interval\': 1, \'max_held\': 4, \'product\': \'EGG\'}, \'COW\': {\'cost\': 400, \'structure\': \'PASTURE\', \'first_yield_day\': 8, \'interval\': 2, \'max_held\': 6, \'product\': \'MILK\'}, \'SHEEP\': {\'cost\': 500, \'structure\': \'PASTURE\', \'first_yield_day\': 6, \'interval\': 3, \'max_held\': 6, \'product\': \'WOOL\'}}\nPRODUCTS = [\'WHEAT\', \'CARROT\', \'TOMATO\', \'STRAWBERRY\', \'MELON\', \'EGG\', \'MILK\', \'WOOL\', \'FERTILIZER\']\nMARKET_I0 = 10000\nPRICE_FLOOR = 1\nMARKET_PARAMS = {\'WHEAT\': {\'base\': 25, \'I0\': MARKET_I0, \'T\': 400, \'below_func\': \'sqrt\', \'below_target\': 0.8, \'above_func\': \'log\', \'above_target\': 0.2}, \'CARROT\': {\'base\': 35, \'I0\': MARKET_I0, \'T\': 450, \'below_func\': \'hinge\', \'below_target\': 1.0, \'above_func\': \'sqrt\', \'above_target\': 0.7}, \'TOMATO\': {\'base\': 60, \'I0\': MARKET_I0, \'T\': 200, \'below_func\': \'hinge\', \'below_target\': 0.4, \'above_func\': \'sqrt\', \'above_target\': 0.6}, \'STRAWBERRY\': {\'base\': 120, \'I0\': MARKET_I0, \'T\': 100, \'below_func\': \'sqrt\', \'below_target\': 0.7, \'above_func\': \'linear\', \'above_target\': 1.6}, \'MELON\': {\'base\': 250, \'I0\': MARKET_I0, \'T\': 300, \'below_func\': \'log\', \'below_target\': 0.2, \'above_func\': \'sq\', \'above_target\': 3.6}, \'EGG\': {\'base\': 50, \'I0\': MARKET_I0, \'T\': 332, \'below_func\': \'hinge\', \'below_target\': 0.4, \'above_func\': \'log\', \'above_target\': 0.2}, \'MILK\': {\'base\': 160, \'I0\': MARKET_I0, \'T\': 122, \'below_func\': \'sqrt\', \'below_target\': 0.6, \'above_func\': \'linear\', \'above_target\': 1.6}, \'WOOL\': {\'base\': 200, \'I0\': MARKET_I0, \'T\': 105, \'below_func\': \'log\', \'below_target\': 0.2, \'above_func\': \'sq\', \'above_target\': 3.2}, \'FERTILIZER\': {\'base\': 100, \'I0\': MARKET_I0, \'T\': 200, \'below_func\': \'linear\', \'below_target\': 0.4, \'above_func\': \'linear\', \'above_target\': 0.4}}\nHINGE_GAIN = 8.0\n\ndef _shape(func, x, T=None):\n    x = max(0.0, x)\n    if func == \'linear\':\n        return x\n    if func == \'sq\':\n        return x * x\n    if func == \'sqrt\':\n        return math.sqrt(x)\n    if func == \'log\':\n        return math.log(1.0 + x)\n    if func == \'log10\':\n        return math.log10(1.0 + x)\n    if func == \'hinge\':\n        if not T or T <= 0:\n            return x\n        u = x / T\n        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2\n    return x\n\ndef _resolve_market_params(overrides):\n    """Merge per-resource overrides onto MARKET_PARAMS defaults (sparse)."""\n    resolved = {item: dict(p) for item, p in MARKET_PARAMS.items()}\n    if not overrides:\n        return resolved\n    for item, patch in overrides.items():\n        if item in resolved and isinstance(patch, dict):\n            resolved[item].update(patch)\n    return resolved\nFARMER_MOVES = {\'NORTH\': (0, -1), \'SOUTH\': (0, 1), \'EAST\': (1, 0), \'WEST\': (-1, 0)}\nLAND_ORDER = [\'NE\', \'SW\', \'SE\']\nLAND_PRICES = [1000, 2000, 4000]\nFARM_HAND_COST_MULT = 1\nSHOPS = {\'BAKERY\': [\'EGG\', \'WHEAT\'], \'PIZZA_SHOP\': [\'MILK\', \'TOMATO\', \'WHEAT\'], \'BRUNCH_SPOT\': [\'EGG\', \'WHEAT\', \'STRAWBERRY\'], \'YARN_STORE\': [\'WOOL\'], \'ICE_CREAM_SHOP\': [\'STRAWBERRY\', \'MILK\', \'WHEAT\'], \'PET_CAFE\': [\'CARROT\'], \'SMOOTHIE_SHOP\': [\'STRAWBERRY\', \'MILK\'], \'FARMERS_MARKET\': [\'WHEAT\', \'CARROT\', \'TOMATO\', \'STRAWBERRY\']}\nTOWN_CENTER_PRODUCTS = [p for p in PRODUCTS if p != \'FERTILIZER\']\nMAX_SHOP_INSTANCES = 8\n\ndef get(d, key, default):\n    if isinstance(d, dict):\n        return d.get(key, default)\n    return getattr(d, key, default)\n\ndef _quadrant_of(x, y, board_size):\n    half = board_size // 2\n    return (\'N\' if y < half else \'S\') + (\'W\' if x < half else \'E\')\n\ndef _shed_access_tiles(board_size):\n    """Four inner-corner tiles around the shed, in NWSE order."""\n    half = board_size // 2\n    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]\n\ndef _is_shed_adjacent(pos, board_size):\n    return tuple(pos) in {(x, y) for x, y in _shed_access_tiles(board_size)}\n\ndef _new_farm(board_size, starting_money):\n    return {\'money\': float(starting_money), \'tiles\': [[_initial_tile(x, y, board_size) for x in range(board_size)] for y in range(board_size)], \'farmer\': list(_default_spawn(board_size)), \'hands\': [], \'unlocked_quadrants\': [\'NW\'], \'hires_today\': 0}\n\ndef _initial_tile(x, y, board_size):\n    return None if _quadrant_of(x, y, board_size) == \'NW\' else \'LOCKED\'\n\ndef _default_spawn(board_size):\n    """First free shed-access tile, NWSE preference."""\n    for tile in _shed_access_tiles(board_size):\n        if _quadrant_of(tile[0], tile[1], board_size) == \'NW\':\n            return tile\n    return (0, 0)\n\ndef _new_private():\n    return {\'shed\': {item: 0 for item in PRODUCTS + list(ANIMALS)}, \'seeds\': {crop: 0 for crop in CROPS}, \'inventories\': [{}]}\n\ndef _new_market(params=None):\n    params = params or MARKET_PARAMS\n    inv = {item: params[item][\'I0\'] for item in PRODUCTS}\n    prices = {item: params[item][\'base\'] for item in PRODUCTS}\n    market = {\'inventory\': inv, \'prices\': prices}\n    if params is not MARKET_PARAMS:\n        market[\'params\'] = params\n    return market\n\ndef _new_town():\n    return {\'unlocked_shops\': []}\n\ndef market_price(item, inventory, params=None):\n    """Floor at PRICE_FLOOR."""\n    p = (params or MARKET_PARAMS)[item]\n    base = p[\'base\']\n    I0 = p[\'I0\']\n    T = p[\'T\']\n    if inventory < I0:\n        f = p[\'below_func\']\n        amp = p[\'below_target\'] * base / _shape(f, T, T)\n        price = base + amp * _shape(f, I0 - inventory, T)\n    else:\n        f = p[\'above_func\']\n        amp = p[\'above_target\'] * base / _shape(f, T, T)\n        price = base - amp * _shape(f, inventory - I0, T)\n    return max(PRICE_FLOOR, int(round(price)))\n\ndef _refresh_prices(market):\n    params = market.get(\'params\')\n    for item in PRODUCTS:\n        market[\'prices\'][item] = market_price(item, market[\'inventory\'][item], params)\n\ndef _new_plant(crop, day, turns_per_day):\n    cd = CROPS[crop]\n    return {\'kind\': \'PLANT\', \'crop\': crop, \'planted_day\': day, \'watered_today\': False, \'consecutive_unwatered\': 1, \'yield_units\': 0 if cd[\'ongoing\'] else 1, \'max_lifespan_step\': -1 if cd[\'ongoing\'] else (day + cd[\'max_yield_day\'] + 1) * turns_per_day, \'fertilized_until_day\': -1}\n\ndef _new_animal(animal, day):\n    a = ANIMALS[animal]\n    return {\'kind\': a[\'structure\'], \'animal\': animal, \'placed_day\': day, \'yield_units\': 0, \'consecutive_unfed\': 0, \'fed_today\': False, \'cared_today\': False, \'fertilizer_available\': False, \'pending_care_bonus\': 0}\n\ndef _farmer_position(farm, idx):\n    """idx 0 = main farmer, 1+ = hand index."""\n    if idx == 0:\n        return farm[\'farmer\']\n    return farm[\'hands\'][idx - 1] if idx - 1 < len(farm[\'hands\']) else None\n\ndef _set_farmer_position(farm, idx, pos):\n    if idx == 0:\n        farm[\'farmer\'] = list(pos)\n    else:\n        farm[\'hands\'][idx - 1] = list(pos)\n\ndef _farmer_inventory(private, idx):\n    """Inventories list is [main_farmer, *hands]; grow it if idx is past the end."""\n    while len(private[\'inventories\']) <= idx:\n        private[\'inventories\'].append({})\n    return private[\'inventories\'][idx]\n\ndef _inv_add(inv, item, n=1):\n    inv[item] = inv.get(item, 0) + n\n\ndef _inv_take(inv, item, n=1):\n    if inv.get(item, 0) < n:\n        return False\n    inv[item] -= n\n    if inv[item] == 0:\n        del inv[item]\n    return True\n\ndef _apply_unit_action(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):\n    """Process one farmer/hand\'s action. Invalid / illegal actions are silent no-ops."""\n    if not isinstance(action, list) or not action:\n        return\n    op = action[0]\n    pos = _farmer_position(farm, idx)\n    if pos is None:\n        return\n    fx, fy = (pos[0], pos[1])\n    inv = _farmer_inventory(private, idx)\n    if op in FARMER_MOVES:\n        dx, dy = FARMER_MOVES[op]\n        nx, ny = (fx + dx, fy + dy)\n        if not (0 <= nx < board_size and 0 <= ny < board_size):\n            return\n        _set_farmer_position(farm, idx, (nx, ny))\n        return\n    if op == \'PASS\':\n        return\n    tile = farm[\'tiles\'][fy][fx]\n    if op == \'DROP\':\n        if not _is_shed_adjacent((fx, fy), board_size):\n            return\n        shed = private[\'shed\']\n        for item, n in list(inv.items()):\n            if n <= 0:\n                del inv[item]\n                continue\n            room = max(0, shed_capacity - sum(shed.values()))\n            take = min(n, room)\n            if take > 0:\n                shed[item] = shed.get(item, 0) + take\n            del inv[item]\n        return\n    if op == \'PICKUP\':\n        if not _is_shed_adjacent((fx, fy), board_size):\n            return\n        if len(action) < 2:\n            return\n        item = action[1]\n        n = int(action[2]) if len(action) >= 3 else 1\n        if n <= 0:\n            return\n        available = private[\'shed\'].get(item, 0)\n        n = min(n, available)\n        if n <= 0:\n            return\n        private[\'shed\'][item] -= n\n        _inv_add(inv, item, n)\n        return\n    if op == \'PLACE\':\n        if len(action) < 2:\n            return\n        item = action[1]\n        if item in ANIMALS and isinstance(tile, dict) and (tile.get(\'kind\') == ANIMALS[item][\'structure\']) and (\'animal\' not in tile):\n            if _inv_take(inv, item, 1):\n                farm[\'tiles\'][fy][fx] = _new_animal(item, day)\n            return\n        if _is_shed_adjacent((fx, fy), board_size):\n            n = int(action[2]) if len(action) >= 3 else 1\n            if n <= 0:\n                return\n            n = min(n, inv.get(item, 0))\n            if n <= 0:\n                return\n            current = sum(private[\'shed\'].values())\n            room = max(0, shed_capacity - current)\n            n = min(n, room)\n            if n <= 0:\n                return\n            inv[item] -= n\n            if inv[item] == 0:\n                del inv[item]\n            private[\'shed\'][item] = private[\'shed\'].get(item, 0) + n\n        return\n    if tile == \'LOCKED\':\n        return\n    if op == \'PLANT\':\n        if len(action) < 2:\n            return\n        crop = action[1]\n        if crop not in CROPS:\n            return\n        if tile is not None:\n            return\n        if private[\'seeds\'].get(crop, 0) <= 0:\n            return\n        private[\'seeds\'][crop] -= 1\n        farm[\'tiles\'][fy][fx] = _new_plant(crop, day, turns_per_day)\n        return\n    if op == \'WATER\':\n        if not (isinstance(tile, dict) and tile.get(\'kind\') == \'PLANT\'):\n            return\n        if tile[\'watered_today\']:\n            return\n        tile[\'watered_today\'] = True\n        crop_data = CROPS[tile[\'crop\']]\n        if not crop_data[\'ongoing\']:\n            age_days = day - tile[\'planted_day\']\n            window_start = (crop_data[\'max_yield_day\'] + 1) // 2\n            if window_start <= age_days <= crop_data[\'max_yield_day\']:\n                bonus = 2 if tile[\'fertilized_until_day\'] >= day else 1\n                tile[\'yield_units\'] = min(crop_data[\'max_yield\'], tile[\'yield_units\'] + bonus)\n        return\n    if op == \'HARVEST\':\n        if not isinstance(tile, dict):\n            return\n        if tile.get(\'yield_units\', 0) <= 0:\n            return\n        if tile.get(\'kind\') == \'PLANT\':\n            crop_data = CROPS[tile[\'crop\']]\n            if day - tile[\'planted_day\'] < crop_data[\'first_yield_day\']:\n                if crop_data[\'ongoing\']:\n                    print(f"WARNING: HARVEST on immature ongoing {tile[\'crop\']} (planted day {tile[\'planted_day\']}, current day {day}, first_yield_day {crop_data[\'first_yield_day\']}, yield_units {tile[\'yield_units\']}); should never happen")\n                return\n            units = tile[\'yield_units\']\n            tile[\'yield_units\'] = 0\n            _inv_add(inv, tile[\'crop\'], units)\n            if not crop_data[\'ongoing\']:\n                farm[\'tiles\'][fy][fx] = None\n        elif \'animal\' in tile:\n            units = tile[\'yield_units\']\n            tile[\'yield_units\'] = 0\n            _inv_add(inv, ANIMALS[tile[\'animal\']][\'product\'], units)\n        return\n    if op == \'FERTILIZE\':\n        if not (isinstance(tile, dict) and tile.get(\'kind\') == \'PLANT\'):\n            return\n        if not _inv_take(inv, \'FERTILIZER\', 1):\n            return\n        tile[\'fertilized_until_day\'] = max(tile.get(\'fertilized_until_day\', -1), day + 2)\n        return\n    if op == \'DIG\':\n        if tile is None:\n            return\n        if isinstance(tile, dict) and \'animal\' in tile:\n            return\n        farm[\'tiles\'][fy][fx] = None\n        return\n    if op == \'BUILD_COOP\':\n        if tile is not None:\n            return\n        farm[\'tiles\'][fy][fx] = {\'kind\': \'COOP\'}\n        return\n    if op == \'BUILD_PASTURE\':\n        if tile is not None:\n            return\n        farm[\'tiles\'][fy][fx] = {\'kind\': \'PASTURE\'}\n        return\n    if op == \'FEED\':\n        if not (isinstance(tile, dict) and \'animal\' in tile):\n            return\n        if tile[\'fed_today\']:\n            return\n        if not _inv_take(inv, \'WHEAT\', 1):\n            return\n        tile[\'fed_today\'] = True\n        return\n    if op == \'COLLECT_FERTILIZER\':\n        if not (isinstance(tile, dict) and \'animal\' in tile):\n            return\n        if not tile[\'fertilizer_available\']:\n            return\n        tile[\'fertilizer_available\'] = False\n        _inv_add(inv, \'FERTILIZER\', 1)\n        return\n    if op == \'CARE\':\n        if not (isinstance(tile, dict) and \'animal\' in tile):\n            return\n        if tile[\'cared_today\']:\n            return\n        tile[\'cared_today\'] = True\n        return\n\ndef _spawn_hand(farm, board_size):\n    """First free shed-access tile (NWSE order); ties broken by min occupancy."""\n    occupants = {tile: 0 for tile in _shed_access_tiles(board_size)}\n    all_pos = [tuple(farm[\'farmer\'])] + [tuple(p) for p in farm[\'hands\']]\n    for pos in all_pos:\n        if pos in occupants:\n            occupants[pos] += 1\n    best = sorted(occupants.items(), key=lambda kv: (kv[1], _shed_access_tiles(board_size).index(kv[0])))\n    return list(best[0][0])\n\ndef _process_market(state, env):\n    """Per-unit lockstep: at each step, quote both players\' current-unit prices, then commit both."""\n    obs0 = state[0].observation\n    market = obs0.market\n    farms = obs0.farms\n    privates = [s.observation.private for s in state]\n    board_size = int(get(env.configuration, \'boardSize\', 10))\n    max_orders = max(1, int(get(env.configuration, \'maxMarketOrdersPerTurn\', 10)))\n    hire_mult = int(get(env.configuration, \'farmHandCostMult\', FARM_HAND_COST_MULT))\n    shed_capacity = int(get(env.configuration, \'shedCapacity\', 100))\n    queues = []\n    for s in state:\n        action = s.action if isinstance(s.action, dict) else {}\n        m = action.get(\'market\', []) if isinstance(action, dict) else []\n        q = list(m) if isinstance(m, list) else []\n        queues.append(q[:max_orders])\n    max_len = max((len(q) for q in queues), default=0)\n    for i in range(max_len):\n        order_states = []\n        for player_id, q in enumerate(queues):\n            ostate = None\n            if i < len(q):\n                ostate = _parse_order(q[i])\n            order_states.append(ostate)\n        for player_id, ostate in enumerate(order_states):\n            if ostate is None:\n                continue\n            op = ostate[\'type\']\n            if op == \'HIRE\':\n                _do_hire(farms[player_id], privates[player_id], board_size, hire_mult)\n                order_states[player_id] = None\n            elif op == \'BUY_LAND\':\n                _do_buy_land(farms[player_id], board_size)\n                order_states[player_id] = None\n        idx_esc = 0\n        while True:\n            idx_esc += 1\n            if idx_esc >= 100000:\n                print(\'WARNING: kaggriculture market loop exceeded 100k iterations; aborting\')\n                break\n            quoted = [None, None]\n            for player_id, ostate in enumerate(order_states):\n                if ostate is None or ostate[\'remaining\'] <= 0:\n                    continue\n                op = ostate[\'type\']\n                item = ostate[\'item\']\n                if op == \'SELL\' and item in PRODUCTS:\n                    quoted[player_id] = (\'SELL\', item, market_price(item, market[\'inventory\'][item], market.get(\'params\')), ostate)\n                elif op == \'BUY_PRODUCT\' and item in (\'WHEAT\', \'FERTILIZER\'):\n                    quoted[player_id] = (\'BUY_PRODUCT\', item, market_price(item, market[\'inventory\'][item] - 1, market.get(\'params\')), ostate)\n                elif op == \'BUY_SEED\' and item in CROPS:\n                    quoted[player_id] = (\'BUY_SEED\', item, CROPS[item][\'seed\'], ostate)\n                elif op == \'BUY_ANIMAL\' and item in ANIMALS:\n                    quoted[player_id] = (\'BUY_ANIMAL\', item, ANIMALS[item][\'cost\'], ostate)\n                else:\n                    order_states[player_id] = None\n            if all((q is None for q in quoted)):\n                break\n            committed_any = False\n            for player_id, q in enumerate(quoted):\n                if q is None:\n                    continue\n                op, item, price, ostate = q\n                ok = _commit_unit(op, item, price, farms[player_id], privates[player_id], market, shed_capacity)\n                if ok:\n                    ostate[\'remaining\'] -= 1\n                    committed_any = True\n                else:\n                    order_states[player_id] = None\n            if not committed_any:\n                break\n        _refresh_prices(market)\n\ndef _parse_order(order):\n    if not isinstance(order, list) or not order:\n        return None\n    op = order[0]\n    if op == \'HIRE\':\n        return {\'type\': \'HIRE\'}\n    if op == \'BUY_LAND\':\n        return {\'type\': \'BUY_LAND\'}\n    if op in (\'BUY_SEED\', \'BUY_PRODUCT\', \'BUY_ANIMAL\', \'SELL\'):\n        if len(order) < 3:\n            return None\n        try:\n            n = int(order[2])\n        except (TypeError, ValueError):\n            return None\n        if n <= 0:\n            return None\n        return {\'type\': op, \'item\': order[1], \'remaining\': n}\n    return None\n\ndef _commit_unit(op, item, price, farm, private, market, shed_capacity=100):\n    if op == \'SELL\':\n        if private[\'shed\'].get(item, 0) <= 0:\n            return False\n        private[\'shed\'][item] -= 1\n        farm[\'money\'] += price\n        if price > 1:\n            market[\'inventory\'][item] += 1\n        return True\n    if op == \'BUY_PRODUCT\':\n        if farm[\'money\'] < price:\n            return False\n        if sum(private[\'shed\'].values()) >= shed_capacity:\n            return False\n        farm[\'money\'] -= price\n        private[\'shed\'][item] = private[\'shed\'].get(item, 0) + 1\n        market[\'inventory\'][item] -= 1\n        return True\n    if op == \'BUY_SEED\':\n        if farm[\'money\'] < price:\n            return False\n        farm[\'money\'] -= price\n        private[\'seeds\'][item] = private[\'seeds\'].get(item, 0) + 1\n        return True\n    if op == \'BUY_ANIMAL\':\n        if farm[\'money\'] < price:\n            return False\n        if sum(private[\'shed\'].values()) >= shed_capacity:\n            return False\n        farm[\'money\'] -= price\n        private[\'shed\'][item] = private[\'shed\'].get(item, 0) + 1\n        return True\n    return False\n\ndef _fib(n):\n    """Indexed so _fib(0)=1, _fib(1)=1, _fib(2)=2, _fib(3)=3, _fib(4)=5..."""\n    a, b = (1, 1)\n    for _ in range(n):\n        a, b = (b, a + b)\n    return a\n\ndef _hire_cost(n_already_today, mult=FARM_HAND_COST_MULT):\n    return mult * _fib(n_already_today)\n\ndef _do_hire(farm, private, board_size, mult=FARM_HAND_COST_MULT):\n    cost = _hire_cost(farm[\'hires_today\'], mult)\n    if farm[\'money\'] < cost:\n        return\n    farm[\'money\'] -= cost\n    farm[\'hires_today\'] += 1\n    farm[\'hands\'].append(_spawn_hand(farm, board_size))\n    private[\'inventories\'].append({})\n\ndef _do_buy_land(farm, board_size):\n    n_unlocked_extra = len(farm[\'unlocked_quadrants\']) - 1\n    if n_unlocked_extra >= len(LAND_ORDER):\n        return\n    cost = LAND_PRICES[n_unlocked_extra]\n    if farm[\'money\'] < cost:\n        return\n    farm[\'money\'] -= cost\n    quadrant = LAND_ORDER[n_unlocked_extra]\n    farm[\'unlocked_quadrants\'].append(quadrant)\n    for y in range(board_size):\n        for x in range(board_size):\n            if _quadrant_of(x, y, board_size) == quadrant and farm[\'tiles\'][y][x] == \'LOCKED\':\n                farm[\'tiles\'][y][x] = None\n\ndef _town_consume(env, state, step):\n    obs0 = state[0].observation\n    market = obs0.market\n    town = obs0.town\n    cfg = env.configuration\n    shop_interval = max(1, int(get(cfg, \'townShopSellInterval\', 4)))\n    center_interval = max(1, int(get(cfg, \'townCenterSellInterval\', 24)))\n    if step % shop_interval == 0:\n        for shop_name in town.get(\'unlocked_shops\', []):\n            products = SHOPS[shop_name]\n            multiplier = 2 if len(products) == 1 else 1\n            for item in products:\n                market[\'inventory\'][item] -= multiplier\n    if step % center_interval == 0:\n        for item in TOWN_CENTER_PRODUCTS:\n            market[\'inventory\'][item] -= 1\n    _refresh_prices(market)\n\ndef _decay_plants(farm, step):\n    board_size = len(farm[\'tiles\'])\n    for y in range(board_size):\n        for x in range(board_size):\n            tile = farm[\'tiles\'][y][x]\n            if not isinstance(tile, dict) or tile.get(\'kind\') != \'PLANT\':\n                continue\n            mls = tile[\'max_lifespan_step\']\n            if mls < 0 or step < mls:\n                continue\n            if (step - mls) % 2 != 0:\n                continue\n            tile[\'yield_units\'] -= 1\n            if tile[\'yield_units\'] <= 0:\n                farm[\'tiles\'][y][x] = {\'kind\': \'WEED\'}\n\ndef _daily_refresh_plants(farm, current_day, turns_per_day):\n    board_size = len(farm[\'tiles\'])\n    next_day = current_day + 1\n    for y in range(board_size):\n        for x in range(board_size):\n            tile = farm[\'tiles\'][y][x]\n            if not isinstance(tile, dict) or tile.get(\'kind\') != \'PLANT\':\n                continue\n            was_watered = tile[\'watered_today\']\n            if was_watered:\n                tile[\'consecutive_unwatered\'] = 0\n            else:\n                tile[\'consecutive_unwatered\'] += 1\n            tile[\'watered_today\'] = False\n            if tile[\'consecutive_unwatered\'] >= 2:\n                farm[\'tiles\'][y][x] = {\'kind\': \'WEED\'}\n                continue\n            cd = CROPS[tile[\'crop\']]\n            if not cd[\'ongoing\']:\n                continue\n            days_since_first = next_day - tile[\'planted_day\'] - cd[\'first_yield_day\']\n            if days_since_first < 0:\n                continue\n            interval = cd[\'interval\']\n            if days_since_first % interval != 0:\n                continue\n            production_count = days_since_first // interval + 1\n            if production_count > cd[\'max_yield\']:\n                continue\n            fertilized = was_watered and tile.get(\'fertilized_until_day\', -1) >= current_day\n            tile[\'yield_units\'] = min(cd[\'max_yield\'], tile[\'yield_units\'] + (2 if fertilized else 1))\n            if production_count == cd[\'max_yield\']:\n                tile[\'max_lifespan_step\'] = (next_day + 1) * turns_per_day\n\ndef _daily_refresh_animals(farm, day):\n    board_size = len(farm[\'tiles\'])\n    next_day = day + 1\n    for y in range(board_size):\n        for x in range(board_size):\n            tile = farm[\'tiles\'][y][x]\n            if not (isinstance(tile, dict) and \'animal\' in tile):\n                continue\n            if tile[\'fed_today\']:\n                tile[\'consecutive_unfed\'] = 0\n            else:\n                tile[\'consecutive_unfed\'] += 1\n            if tile[\'consecutive_unfed\'] >= 2:\n                farm[\'tiles\'][y][x] = {\'kind\': ANIMALS[tile[\'animal\']][\'structure\']}\n                continue\n            a = ANIMALS[tile[\'animal\']]\n            days_since_first = next_day - tile[\'placed_day\'] - a[\'first_yield_day\']\n            if days_since_first >= 0 and days_since_first % a[\'interval\'] == 0:\n                base = 1\n                bonus = tile.pop(\'pending_care_bonus\', 0) if tile[\'fed_today\'] else 0\n                tile[\'yield_units\'] = min(a[\'max_held\'], tile[\'yield_units\'] + base + bonus)\n                tile[\'pending_care_bonus\'] = 0\n            if tile[\'cared_today\'] and tile[\'fed_today\']:\n                tile[\'pending_care_bonus\'] = tile.get(\'pending_care_bonus\', 0) + 1\n            tile[\'fertilizer_available\'] = True\n            tile[\'fed_today\'] = False\n            tile[\'cared_today\'] = False\n\ndef _spawn_weeds(farm, board_size, weed_chance, rng):\n    for y in range(board_size):\n        for x in range(board_size):\n            if farm[\'tiles\'][y][x] is None and rng.random() < weed_chance:\n                farm[\'tiles\'][y][x] = {\'kind\': \'WEED\'}\n\ndef _drop_inventories_to_shed(private, capacity):\n    """Drop every per-farmer inventory into the shed up to `capacity`; overflow is discarded.\n    Seeds are tracked separately in private["seeds"] and don\'t pass through the shed."""\n    shed = private[\'shed\']\n    for inv in private[\'inventories\']:\n        for item, n in list(inv.items()):\n            if n <= 0:\n                del inv[item]\n                continue\n            current = sum((v for k, v in shed.items()))\n            room = max(0, capacity - current)\n            take = min(n, room)\n            if take > 0:\n                shed[item] = shed.get(item, 0) + take\n            del inv[item]\n\ndef _end_of_day(state, env, day):\n    obs0 = state[0].observation\n    cfg = env.configuration\n    board_size = int(get(cfg, \'boardSize\', 10))\n    turns_per_day = max(1, int(get(cfg, \'turnsPerDay\', 24)))\n    weed_chance = float(get(cfg, \'weedSpawnChance\', 0.005))\n    shed_cap = int(get(cfg, \'shedCapacity\', 100))\n    shop_interval = max(1, int(get(cfg, \'townShopUnlockInterval\', 3)))\n    seed = env.info.get(\'seed\', 0)\n    rng = random.Random(seed * 1000003 ^ day)\n    for player_id, farm in enumerate(obs0.farms):\n        private = state[player_id].observation.private\n        _daily_refresh_plants(farm, day, turns_per_day)\n        _daily_refresh_animals(farm, day)\n        _spawn_weeds(farm, board_size, weed_chance, rng)\n        _drop_inventories_to_shed(private, shed_cap)\n        farm[\'farmer\'] = list(_default_spawn(board_size))\n        farm[\'hands\'] = []\n        farm[\'hires_today\'] = 0\n        private[\'inventories\'] = [{}]\n    next_day = day + 1\n    town = obs0.town\n    if next_day > 0 and next_day % shop_interval == 0:\n        if len(town[\'unlocked_shops\']) < MAX_SHOP_INSTANCES:\n            town[\'unlocked_shops\'].append(rng.choice(sorted(SHOPS)))',SIM)
ROLLOUT_CHOICES=[(9, 120), (11, 120), (13, 120)]
import copy
from types import SimpleNamespace

class SimStruct(dict):
    def __getattr__(self,k):return self[k]
    def __setattr__(self,k,v):self[k]=v

def farm_value(obs,prices):
    f=obs['farms'][obs['player']];p=obs['private'];day=obs['day']
    value=f['money']+sum(n*prices.get(c,ANIMALS.get(c,(0,))[0]) for c,n in totals(p).items())
    value+=sum(n*CROPS[c][0]*.5 for c,n in p['seeds'].items())
    for row in f['tiles']:
        for t in row:
            if not isinstance(t,dict):continue
            if 'animal' in t:
                a=ANIMALS[t['animal']]
                value+=a[0]*1.2+t.get('yield_units',0)*prices[a[1]]
                value+=min(4,t.get('pending_care_bonus',0))*prices[a[1]]*.8
                value+=t.get('fertilizer_available',False)*prices['FERTILIZER']*.8
            elif 'crop' in t:
                c=t['crop'];age=day-t['planted_day'];held=t['yield_units']
                if c in ('STRAWBERRY','TOMATO'):
                    future=sum(n for at,n in CROPS[c][2] if at>age and day+at-age<=29)
                    v=held*prices[c]+future*prices[c]*.6
                else:
                    remaining=max(0,CROPS[c][3]-age)
                    v=(held+min(remaining,3))*prices[c]*(.9**remaining)
                value+=v*(.85 if t.get('consecutive_unwatered') else 1)
    value+=sum((1000,2000,4000)[:max(0,len(f['unlocked_quadrants'])-1)])
    return value

def simulate_day(obs,hands,plant):
    global HIRE_OVERRIDE,PLANT_WEIGHT
    HIRE_OVERRIDE=hands;PLANT_WEIGHT=plant
    o=SimStruct(copy.deepcopy(obs));seat=o.player
    private_other=SIM['_new_private']()
    other=SimStruct(o.copy());other.player=1-seat;other.private=private_other
    state=[None,None];state[seat]=SimStruct(observation=o,action={},reward=0,status='ACTIVE')
    state[1-seat]=SimStruct(observation=other,action={},reward=0,status='ACTIVE')
    env=SimpleNamespace(configuration=SimStruct(episodeSteps=720),info={},done=False)
    prices=dict(o.market['prices']);day=o.day
    for hour in range(o.hour,23 if day==29 else 24):
        o.hour=hour;o.step=day*24+hour
        action=policy(o);state[seat].action=action
        for i,a in enumerate([action['farmer']]+action['hands']):
            SIM['_apply_unit_action'](o.farms[seat],o.private,i,a,10,day,24,100)
        SIM['_process_market'](state,env)
        SIM['_town_consume'](env,state,o.step)
        SIM['_decay_plants'](o.farms[seat],o.step)
    if day<29:
        SIM['_daily_refresh_plants'](o.farms[seat],day,24)
        SIM['_daily_refresh_animals'](o.farms[seat],day)
        SIM['_drop_inventories_to_shed'](o.private,100)
        o.day=day+1
    return farm_value(o,prices)

def agent(obs):
    global HIRE_OVERRIDE,PLANT_WEIGHT,OPP_STYLE
    if obs['day']==0 and obs['hour']==0:
        HIRE_OVERRIDE=None;PLANT_WEIGHT=120
    elif obs['hour']==0 and obs['day']>=3:
        saved_style=OPP_STYLE
        choices=[]
        for hands,plant in ROLLOUT_CHOICES:
            value=simulate_day(obs,hands,plant)
            choices.append((value,hands,plant))
        _,HIRE_OVERRIDE,PLANT_WEIGHT=max(choices)
        OPP_STYLE=saved_style
    return policy(obs)
