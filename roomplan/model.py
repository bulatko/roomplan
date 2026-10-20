import base64
import copy
import math
import re
from .research import validate_research_layers
from .scenes import validate_scene_refs

MAX_BLOB = 100 * 1024 * 1024
MAX_TOTAL = 250 * 1024 * 1024

CATALOG = []
def _items(category, kind, names, sizes, colors, mount='floor'):
    for i, name in enumerate(names):
        CATALOG.append(dict(id=f'{kind}-{i+1}', name=name, category=category, kind=kind,
                            size=sizes[i], color=colors[i], mount=mount))

_items('Диваны', 'sofa', ['Диван Linen', 'Диван Moss'], [[2.25,.82,.9],[2.65,.78,1.05]], ['#d4c6ae','#799184'])
_items('Кресла', 'armchair', ['Кресло Shell','Кресло Lounge'], [[.8,.8,.8],[.9,.75,.88]], ['#c19b78','#b2bab2'])
_items('Журнальные столы','coffeeTable',['Стол Pebble','Стол Oak'],[[1.1,.38,.6],[.85,.42,.85]],['#b78d65','#d7b88c'])
_items('Кровати','bed',['Кровать Soft 160','Кровать Soft 180'],[[1.7,.55,2.15],[1.9,.58,2.2]],['#e5dfd4','#b7beb8'])
_items('Хранение','wardrobe',['Шкаф Birch','Шкаф Graphite'],[[1.8,2.3,.6],[1.2,2.2,.6]],['#e7dfcf','#666e70'])
_items('Обеденные столы','diningTable',['Стол Arc','Стол Family'],[[1.4,.75,.85],[1.8,.76,.9]],['#cba276','#c0a88e'])
_items('Стулья','diningChair',['Стул Wish','Стул Form'],[[.46,.8,.5],[.5,.82,.52]],['#b99b78','#8a9b8b'])
_items('Тумбы','cabinet',['Тумба Linear','Тумба Cube'],[[1.8,.48,.4],[.6,.5,.45]],['#b69b7a','#d7cfbf'])
_items('Рабочие столы','desk',['Стол Studio','Стол Compact'],[[1.4,.75,.65],[1.1,.75,.6]],['#c1a480','#d8d3c7'])
_items('Рабочие кресла','officeChair',['Кресло Task','Кресло Mesh'],[[.6,1,.6],[.65,1.1,.65]],['#606e66','#898f8b'])
_items('Холодильники','fridge',['Холодильник Tall','Холодильник Wide'],[[.65,1.9,.65],[.85,1.8,.7]],['#c3c9c7','#767f7e'])
_items('Кухонные модули','stove',['Модуль с плитой','Модуль с духовкой'],[[.6,.9,.6],[.8,.9,.6]],['#d4d0c5','#b6bcb7'])
_items('Мойки','sink',['Мойка Kitchen','Раковина Bath'],[[.8,.9,.6],[.65,.85,.5]],['#d6d4cc','#f3f2ec'])
_items('Санузел','toilet',['Унитаз Compact','Унитаз Oval'],[[.4,.75,.65],[.42,.72,.7]],['#f8f7f2','#edede7'])
_items('Ванны','bathtub',['Ванна Oval','Ванна Straight'],[[.8,.58,1.7],[.75,.6,1.6]],['#f3f1e9','#f8f6ef'])
_items('Растения','plant',['Фикус','Монстера'],[[.55,1.5,.55],[.7,1.1,.7]],['#548365','#6d8c62'])
_items('Телевизоры','tv',['Экран 55','Экран 43'],[[1.23,.72,.1],[.97,.6,.1]],['#283539','#394343'],'surface')
_items('Ковры','rug',['Ковер Sand','Ковер Sage'],[[2.4,.015,1.8],[2.1,.015,1.6]],['#d5cbb5','#b0b7a6'])

def polygon_area(points):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points, points[1:]+points[:1])))/2

def demo_project(height=2.8):
    p = dict(version=1,name='Квартира · Тёплый минимализм',ceilingHeight=height,walls=[],rooms=[],objects=[],pointCloud=[],layers={'architecture':True,'furniture':True,'pointCloud':False,'boxes':False,'research':False},sources=[],assets=[],provenance={'mode':'demo','method':'authored-demo','isSynthetic':True})
    def wall(a,b):
        p['walls'].append(dict(id=f'w{len(p["walls"])+1}',a=list(a),b=list(b),height=height,thickness=.16))
    # 10 x 6.5 m envelope. Deliberate 0.9–1.0 m door openings.
    for a,b in [((0,0),(10,0)),((10,0),(10,6.5)),((10,6.5),(6,6.5)),((5,6.5),(0,6.5)),((0,6.5),(0,0)),
                ((5.8,0),(5.8,3.55)),((5.8,4.5),(5.8,6.5)),((5.8,3.25),(6.45,3.25)),((7.4,3.25),(10,3.25)),
                ((7.7,3.25),(7.7,4.65)),((7.7,5.55),(7.7,6.5))]: wall(a,b)
    p['openings']=[
        dict(id='door1',kind='door',a=[5.8,3.55],b=[5.8,4.5],height=2.1,sillHeight=0),
        dict(id='door2',kind='door',a=[6.45,3.25],b=[7.4,3.25],height=2.1,sillHeight=0),
        dict(id='door3',kind='door',a=[7.7,4.65],b=[7.7,5.55],height=2.1,sillHeight=0),
        dict(id='door4',kind='door',a=[5,6.5],b=[6,6.5],height=2.1,sillHeight=0),
        dict(id='window1',kind='window',wallId='w1',a=[.9,0],b=[4.8,0],height=1.6,sillHeight=.85),
        dict(id='window2',kind='window',wallId='w1',a=[6.6,0],b=[9.3,0],height=1.6,sillHeight=.85),
        dict(id='window3',kind='window',wallId='w5',a=[0,3],b=[0,5.5],height=1.6,sillHeight=.85)]
    for name,poly,color in [('Кухня-гостиная',[[0,0],[5.8,0],[5.8,6.5],[0,6.5]],'#e9dfca'),('Спальня',[[5.8,0],[10,0],[10,3.25],[5.8,3.25]],'#e8d9c8'),('Прихожая',[[5.8,3.25],[7.7,3.25],[7.7,6.5],[5.8,6.5]],'#ded7ca'),('Ванная',[[7.7,3.25],[10,3.25],[10,6.5],[7.7,6.5]],'#e0e5df')]:
        p['rooms'].append(dict(id=f'r{len(p["rooms"])+1}',name=name,polygon=poly,color=color))
    def obj(cid,x,z,rot=0,y=0,**kw):
        c=next(c for c in CATALOG if c['id']==cid)
        p['objects'].append(dict(id=f'o{len(p["objects"])+1}',catalogId=cid,name=c['name'],kind=c['kind'],position=[x,y,z],rotation=rot,size=c['size'][:],color=c['color'],mount=c['mount'],**kw))
    obj('rug-1',2.4,4.2);obj('sofa-1',1,4.15,math.pi/2);obj('coffeeTable-1',2.35,4.1);obj('armchair-1',2.8,5.25,math.pi)
    obj('cabinet-1',5.42,5.4,math.pi/2);obj('tv-1',5.42,5.4,math.pi/2,.48,supportId='o5')
    obj('fridge-1',.55,.5);obj('sink-1',1.4,.4);obj('stove-1',2.15,.4);obj('cabinet-1',3.45,.4,0,0)
    obj('diningTable-1',2.5,1.9);obj('diningChair-1',1.55,1.9,math.pi/2);obj('diningChair-1',3.45,1.9,-math.pi/2);obj('diningChair-1',2.5,2.6,0)
    obj('plant-1',.55,5.8);obj('bed-1',8,1.3);obj('cabinet-2',6.8,.65);obj('cabinet-2',9.2,.65)
    obj('wardrobe-1',9.65,1.8,math.pi/2);obj('plant-2',6.3,2.65);obj('wardrobe-2',6.2,5.3,math.pi/2)
    obj('bathtub-1',9.4,4.35);obj('toilet-1',8.2,3.85);obj('sink-2',8.55,6.12,math.pi)
    return p

def finite(value, label, lo=-1000, hi=1000):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f'{label}: ожидается конечное число в диапазоне {lo}…{hi}')
    return value

def vector(value,n,label):
    if not isinstance(value,list) or len(value)!=n: raise ValueError(f'{label}: требуется {n} координат')
    for v in value: finite(v,label)

def validate_settings(settings):
    """Validate fields consumed by the viewer, retaining optional/extension data."""
    if not isinstance(settings,dict): raise ValueError('settings: ожидается объект')
    for field in ['cutaway','dimensions','roomLabels']:
        if field in settings and not isinstance(settings[field],bool):
            raise ValueError(f'settings.{field}: ожидается логическое значение')
    for field,choices in [('view',('2d','3d')),('walkPathMode',('custom','overview','interior'))]:
        if field in settings and settings[field] not in choices:
            raise ValueError(f'settings.{field}: неизвестное значение')
    if 'cameraFov' in settings:
        fov=finite(settings['cameraFov'],'settings.cameraFov',0,180)
        if not 0<fov<180: raise ValueError('settings.cameraFov: угол должен быть больше 0 и меньше 180')
    # Broad bounds accommodate saved views beyond the UI slider's range while
    # avoiding zero/negative scales and numerically unusable projections.
    if 'orthoZoom' in settings: finite(settings['orthoZoom'],'settings.orthoZoom',1e-6,1e6)
    if 'light' in settings: finite(settings['light'],'settings.light',1e-6,1000)
    def camera_point(point,label):
        if not isinstance(point,dict): raise ValueError(f'{label}: ожидается объект камеры')
        vector(point.get('position'),3,f'{label}.position')
        vector(point.get('target'),3,f'{label}.target')
    for field in ['camera','orthoCamera']:
        if field in settings: camera_point(settings[field],f'settings.{field}')
    if 'walkPath' in settings:
        path=settings['walkPath']
        if not isinstance(path,list) or len(path)>2000: raise ValueError('settings.walkPath: требуется массив не более 2000 точек')
        # Empty and single-point routes are valid drafts; recording validates
        # the minimum point count and collisions separately.
        for i,point in enumerate(path): camera_point(point,f'settings.walkPath[{i}]')

def validate_files(files, max_files=100):
    if not isinstance(files,list) or len(files)>max_files: raise ValueError('Слишком много файлов')
    total=0
    for f in files:
        if not isinstance(f,dict): raise ValueError('Неверное описание файла')
        if not isinstance(f.get('name'),str) or len(f['name'])>300: raise ValueError('Неверное имя файла')
        if not isinstance(f.get('mime'),str) or len(f['mime'])>100: raise ValueError('Неверный MIME файла')
        if 'data' not in f: raise ValueError('Отсутствует содержимое файла data')
        data=f['data']
        if not isinstance(data,str) or len(data)>MAX_BLOB*1.4: raise ValueError('Файл превышает 100 МБ')
        try: raw=base64.b64decode(data,validate=True)
        except Exception as e: raise ValueError('Некорректный base64 файла') from e
        total+=len(raw)
        if len(raw)>MAX_BLOB or total>MAX_TOTAL: raise ValueError('Превышен лимит файлов проекта (250 МБ)')
    return files

def validate_project(project):
    if not isinstance(project,dict) or project.get('version')!=1: raise ValueError('Поддерживается версия проекта 1')
    p=copy.deepcopy(project); warnings=[]
    def check_json(value,depth=0):
        if depth>30: raise ValueError('Слишком глубокая структура метаданных')
        if isinstance(value,float) and not math.isfinite(value): raise ValueError('Неконечное число в проекте')
        if isinstance(value,dict):
            for v in value.values(): check_json(v,depth+1)
        elif isinstance(value,list):
            for v in value: check_json(v,depth+1)
    check_json(p)
    if 'settings' in p: validate_settings(p['settings'])
    if not isinstance(p.get('name'),str) or len(p['name'])>200: raise ValueError('Неверное название проекта')
    finite(p.get('ceilingHeight'),'Высота потолка',1.5,10)
    ids=set()
    for field,limit in [('walls',2000),('rooms',500),('objects',2000),('pointCloud',100000),('sources',100),('assets',100)]:
        if not isinstance(p.get(field,[]),list) or len(p.get(field,[]))>limit: raise ValueError(f'Неверный массив {field}; максимум {limit}')
        p.setdefault(field,[])
    for collection in ['walls','rooms','objects']:
        for item in p[collection]:
            if not isinstance(item,dict): raise ValueError(f'{collection}: неверный объект')
            ident=item.get('id')
            if not isinstance(ident,str) or len(ident)>100 or ident in ids: raise ValueError('Идентификаторы должны быть уникальными строками')
            ids.add(ident)
            if collection in ['rooms','objects'] and any(key not in item for key in ['name','color']): raise ValueError(f'{collection}: обязательны name и color')
            if 'name' in item and (not isinstance(item['name'],str) or len(item['name'])>200): raise ValueError('Неверное имя элемента')
            if 'color' in item and (not isinstance(item['color'],str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',item['color'])): raise ValueError('Цвет должен быть #RRGGBB')
    for w in p['walls']:
        vector(w.get('a'),2,'Начало стены');vector(w.get('b'),2,'Конец стены')
        if math.dist(w['a'],w['b'])<.01: raise ValueError('Длина стены меньше 1 см')
        finite(w.get('height'),'Высота стены',.1,10);finite(w.get('thickness'),'Толщина стены',.01,2)
    for r in p['rooms']:
        if not isinstance(r.get('polygon'),list) or not 3<=len(r['polygon'])<=1000: raise ValueError('Неверный многоугольник комнаты')
        for v in r['polygon']: vector(v,2,'Вершина комнаты')
        if polygon_area(r['polygon'])<.01: raise ValueError('Нулевая площадь комнаты')
    for o in p['objects']:
        vector(o.get('position'),3,'Положение объекта');vector(o.get('size'),3,'Размер объекта')
        for v in o['size']: finite(v,'Размер объекта',.001,100)
        finite(o.get('rotation'),'Поворот',-1000,1000)
        if o.get('mount') not in ['floor','wall','surface']: raise ValueError('Неизвестный тип установки')
        if o['position'][1]<-.001: raise ValueError('Объект ниже пола')
        if o['position'][1]+o['size'][1]>p['ceilingHeight']+.01: warnings.append(f'{o.get("name",o["id"])} выше потолка')
    for point in p['pointCloud']: vector(point,3,'Точка облака')
    if not isinstance(p.get('openings',[]),list) or len(p.get('openings',[]))>2000: raise ValueError('Неверный массив проёмов')
    for o in p.get('openings',[]):
        if not isinstance(o,dict): raise ValueError('Неверное описание проёма')
        vector(o.get('a'),2,'Начало проёма');vector(o.get('b'),2,'Конец проёма')
        if o.get('kind') not in ['door','window']: raise ValueError('Неизвестный тип проёма')
        finite(o.get('height'),'Высота проёма',.1,10);finite(o.get('sillHeight'),'Подоконник',0,10)
    validate_files(p['sources']);validate_files(p['assets'])
    for kind in ['sources','assets']:
        file_ids=set()
        for f in p[kind]:
            fid=f.get('id')
            if not isinstance(fid,str) or not fid or len(fid)>100 or fid in file_ids: raise ValueError(f'{kind}: требуется уникальный строковый id')
            file_ids.add(fid)
    if sum(len(f['data']) for k in ['sources','assets'] for f in p[k])>MAX_TOTAL*1.34: raise ValueError('Проект превышает 250 МБ')
    asset_ids={f.get('id') for f in p['assets']}
    validate_research_layers(p.get('researchLayers', []), {f['id'] for f in p['sources']}, asset_ids)
    if 'scenes' in p: validate_scene_refs(p['scenes'])
    for o in p['objects']:
        if o.get('assetId') and o['assetId'] not in asset_ids: raise ValueError('У объекта отсутствует ресурс assetId')
    if not isinstance(p.get('layers',{}),dict) or not isinstance(p.get('provenance',{}),dict): raise ValueError('Неверные метаданные проекта')
    return p,list(dict.fromkeys(warnings))
