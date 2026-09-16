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
        if day<=1 and p in wheat|melon:
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
                tasks.append((p,['PLANT',c],100 if day==0 else 120,None));available[c]-=1
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
    needs={}
    for pos,c in crops.items():
        t=tile(f,pos)
        if (t is None or (isinstance(t,dict) and t.get('kind')=='WEED')) and hour<17:needs[c]=needs.get(c,0)+1
    seed_order=lambda kv:(0 if OPP_STYLE=='V16' and day==0 and kv[0]=='MELON' else 1,CROPS[kv[0]][0])
    for c,n in sorted(needs.items(),key=seed_order):
        n=min(max(0,n-p['seeds'].get(c,0)),int(max(0,cash-80)//CROPS[c][0]))
        if n and len(orders)<10:orders.append(['BUY_SEED',c,n]);cash-=n*CROPS[c][0]
    for a in ('COW','SHEEP','GOOSE'):
        buy_deadline=17
        if desired[a]>counts[a] and day<=buy_deadline and cash>ANIMALS[a][0]+50 and len(orders)<10:
            orders.append(['BUY_ANIMAL',a,1]);cash-=ANIMALS[a][0]
    fert_need=0
    for row in f['tiles']:
        for t in row:
            if isinstance(t,dict) and 'crop' in t and fert_value(t,day,prices)>10:fert_need+=1
    target=min(24,fert_need+3) if fert_need else 0
    if total.get('FERTILIZER',0)<target and day<29 and len(orders)<10:
        n=min(target-total.get('FERTILIZER',0),int(max(0,cash-100)//(prices['FERTILIZER']+3)))
        if n:orders.append(['BUY_PRODUCT','FERTILIZER',n])
    return orders[:10]

def agent(obs):
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
