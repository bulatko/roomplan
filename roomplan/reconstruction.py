"""CPU baseline for clean axis-aligned raster plans; no learned inference."""
import base64
import io
import math
import uuid
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from .model import demo_project, finite, validate_files

def _runs(row, minimum):
    d=np.diff(np.r_[False,row,False].astype(np.int8))
    return [(int(a),int(b-1)) for a,b in zip(np.where(d==1)[0],np.where(d==-1)[0]) if b-a>=minimum]

def _lines(mask, minimum):
    groups=[]
    for y,row in enumerate(mask):
        for a,b in _runs(row,minimum):
            match=None
            for g in reversed(groups[-250:]):
                if y-g[3]>2: continue
                if abs(a-g[0])<=5 and abs(b-g[1])<=5:
                    match=g;break
            if match is None: groups.append([a,b,y,y,1])
            else:
                n=match[4];match[0]=(match[0]*n+a)/(n+1);match[1]=(match[1]*n+b)/(n+1);match[3]=y;match[4]+=1
    # At a T/L junction a thick stroke changes length across adjacent scan rows.
    # Merge touching strips with nearly identical spans, rather than emitting
    # two parallel walls. Never bridge a longitudinal door gap.
    merged=[]
    for g in groups:
        if g[4]<2: continue
        for m in merged:
            overlap=max(0,min(g[1],m[1])-max(g[0],m[0]))
            shortest=min(g[1]-g[0],m[1]-m[0])
            transverse_gap=max(g[2],m[2])-min(g[3],m[3])
            tolerance=max(20,2*(g[3]-g[2]+m[3]-m[2]))
            if transverse_gap<=2 and overlap>=shortest*.9 and max(abs(g[0]-m[0]),abs(g[1]-m[1]))<=tolerance:
                m[:]=[min(g[0],m[0]),max(g[1],m[1]),min(g[2],m[2]),max(g[3],m[3]),g[4]+m[4]]
                break
        else: merged.append(g[:])
    return merged

def join_axis_walls(walls):
    """Join stroke ends at perpendicular centre lines, within stroke thickness.

    Raster run endpoints lie at the outside edge of intersecting thick strokes.
    Keeping those raw ends creates visually overlapping, topologically unjoined
    walls. Only perpendicular intersections near an existing endpoint qualify;
    longitudinal gaps (doors) are never closed by this operation.
    """
    original=[dict(w,a=w['a'][:],b=w['b'][:]) for w in walls]
    for wall,raw in zip(walls,original):
        horizontal=abs(raw['a'][1]-raw['b'][1])<1e-6
        axis=0 if horizontal else 1
        for key in ('a','b'):
            point=raw[key]
            candidates=[]
            for other in original:
                other_horizontal=abs(other['a'][1]-other['b'][1])<1e-6
                if other_horizontal==horizontal: continue
                tolerance=(raw['thickness']+other['thickness'])/2+.015
                intersection=point[:]
                intersection[axis]=other['a'][axis]
                distance=abs(intersection[axis]-point[axis])
                lo=min(other['a'][1-axis],other['b'][1-axis])-tolerance
                hi=max(other['a'][1-axis],other['b'][1-axis])+tolerance
                if distance<=tolerance and lo<=point[1-axis]<=hi:
                    candidates.append((distance,intersection))
            if candidates: wall[key]=min(candidates,key=lambda item:item[0])[1]


def recognize_plan(file,height,width):
    try:
        im=Image.open(io.BytesIO(base64.b64decode(file['data'])))
        if im.width*im.height>40_000_000: raise ValueError('Изображение превышает 40 мегапикселей')
        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1500,1500))
        gray=ImageOps.grayscale(im)
        # Small median suppresses JPEG noise; threshold rejects pale decorative fills.
        arr=np.asarray(gray.filter(ImageFilter.MedianFilter(3)))
    except ValueError: raise
    except Exception as e: raise ValueError('Не удалось прочитать изображение плана; используйте PNG или JPEG') from e
    threshold=min(175,max(70,float(np.percentile(arr,15))))
    mask=arr<threshold
    minimum=max(20,int(min(mask.shape)*.055))
    hs=_lines(mask,minimum);vs=_lines(mask.T,minimum)
    lines=[(a,(c+d)/2,b,(c+d)/2,max(2,d-c+1)) for a,b,c,d,n in hs]+[((c+d)/2,a,(c+d)/2,b,max(2,d-c+1)) for a,b,c,d,n in vs]
    if len(lines)<4: raise ValueError('Не найдено достаточно стен. Нужен контрастный план сверху без перспективы: тёмные горизонтальные и вертикальные стены на светлом фоне.')
    if len(lines)>500: raise ValueError('Слишком много линий: обрежьте легенду и мебель, используйте более чистый план')
    x0=min(min(l[0],l[2]) for l in lines);x1=max(max(l[0],l[2]) for l in lines)
    z0=min(min(l[1],l[3]) for l in lines);z1=max(max(l[1],l[3]) for l in lines)
    if x1-x0<20 or z1-z0<20: raise ValueError('Вырожденная геометрия плана')
    scale=width/(x1-x0);depth=(z1-z0)*scale
    p=dict(version=1,name=file['name'].rsplit('.',1)[0],ceilingHeight=height,walls=[],rooms=[dict(id='r1',name='Область плана · проверить',polygon=[[0,0],[width,0],[width,depth],[0,depth]],color='#e8decd')],objects=[],pointCloud=[],sources=[],assets=[],layers={'architecture':True,'furniture':True,'pointCloud':False,'boxes':False,'research':False},provenance={'mode':'plan','method':'threshold-axis-line-baseline-v1','isSynthetic':False,'calibration':{'widthMeters':width,'pixelWidth':x1-x0,'metersPerPixel':scale},'imageSize':list(im.size),'roomBoundary':'bounding-envelope-estimate'})
    for i,(a,b,c,d,t) in enumerate(lines):
        p['walls'].append(dict(id=f'w{i+1}',a=[round((a-x0)*scale,4),round((b-z0)*scale,4)],b=[round((c-x0)*scale,4),round((d-z0)*scale,4)],height=height,thickness=round(max(.04,min(.5,t*scale)),4)))
    join_axis_walls(p['walls'])
    vertices=[wall[key] for wall in p['walls'] for key in ('a','b')]
    xmin,xmax=min(v[0] for v in vertices),max(v[0] for v in vertices)
    zmin,zmax=min(v[1] for v in vertices),max(v[1] for v in vertices)
    # Calibrate the joined centre-line envelope, not the outer raster stroke.
    # This keeps the requested metric width exact after removing stroke overlap.
    correction=width/(xmax-xmin)
    for wall in p['walls']:
        for key in ('a','b'):
            wall[key]=[round((wall[key][0]-xmin)*correction,4),round((wall[key][1]-zmin)*correction,4)]
        wall['thickness']=round(wall['thickness']*correction,4)
    calibrated_depth=round((zmax-zmin)*correction,4)
    p['rooms'][0]['polygon']=[[0,0],[width,0],[width,calibrated_depth],[0,calibrated_depth]]
    p['provenance']['calibration'].update(rawRasterPixelWidth=x1-x0,pixelWidth=width/(scale*correction),metersPerPixel=scale*correction,reference='joined-outer-wall-centrelines')
    p['provenance']['topology']='perpendicular-centreline-junctions'
    warnings=['Базовый алгоритм: пороговая обработка и поиск осевых линий, без нейросети. Поддерживаются чистые прямоугольные планы без перспективы.', 'Линии мебели, размеров и текста могут стать ложными стенами. Диагональные и кривые стены не поддерживаются; проверьте и исправьте результат.', 'Ширина откалибрована между осями крайних стен; высота стен введена пользователем. Контур пола — оценка по габаритам, комнаты и двери автоматически не распознаются.']
    return p,warnings

def synthetic_cloud(p):
    rng=np.random.default_rng(17);points=[]
    for w in p['walls']:
        a=np.array(w['a']);b=np.array(w['b'])
        for t in np.linspace(0,1,max(3,int(np.linalg.norm(b-a)*13))):
            for y in np.linspace(.08,w['height'],12):
                v=a+(b-a)*t+rng.normal(0,.012,2);points.append([round(float(v[0]),3),round(float(y),3),round(float(v[1]),3)])
    return points

def reconstruct(payload):
    mode=payload.get('mode');files=validate_files(payload.get('files',[]),30)
    if not files: raise ValueError('Добавьте исходный файл')
    height=finite(payload.get('ceilingHeight',2.8),'Высота потолка',1.5,10)
    if mode=='plan':
        width=finite(payload.get('widthMeters',10),'Полная ширина плана',1,100)
        p,warnings=recognize_plan(files[0],height,width)
    elif mode in ['video','photos']:
        p=demo_project(height);p['name']='Демонстрация по '+('видео' if mode=='video' else 'фотографиям')
        p['pointCloud']=synthetic_cloud(p)
        p['layers']['pointCloud']=True
        p['provenance']={'mode':mode,'method':'deterministic-demo-stub-v1','isSynthetic':True,'inputUsedForGeometry':False}
        warnings=['ДЕМОНСТРАЦИОННАЯ ЗАГЛУШКА: геометрия и облако точек синтетические. Содержимое фотографий/видео не анализируется. Исходные файлы сохранены для будущего подключения SfM / SLAM / распознавания.']
        if mode=='video': warnings.append('Настоящий план по видео строит внешний сервис реконструкции: POST /api/reconstruct/video при заданных ROOMPLAN_VIDEO_API_URL и ROOMPLAN_VIDEO_API_TOKEN.')
    else: raise ValueError('Неизвестный режим реконструкции')
    p['sources']=[dict(f,id=f.get('id',str(uuid.uuid4()))) for f in files]
    return p,warnings
