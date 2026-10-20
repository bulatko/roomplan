"""Video -> editable plan through the roomplan-app HTTP API.

The GPU pipeline (beam_rec reconstruction -> wallfield P4 walls -> TUN3D objects/openings) runs
as a separate service; this module only uploads the video, relays job status and converts the
finished result into a project. No local inference, no synthetic fallback.

Configuration (environment, or `.env` next to server.py):
  ROOMPLAN_VIDEO_API_URL      service root, e.g. http://host:8802
  ROOMPLAN_VIDEO_API_TOKEN    user token (header X-Token)
  ROOMPLAN_VIDEO_API_OPTIONS  optional JSON job options, e.g. {"beam_rec":{"target_frames":200}}

Service API: POST /api/jobs (multipart: video, include_cloud, formats, options) -> job;
GET /api/jobs/<id> -> job {status: queued|running|done|failed, stage, message, error, files};
GET /api/jobs/<id>/files/<name>. Result world: metres, Z up, floor near z=0.
"""
import base64
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
import numpy as np
from .model import CATALOG, validate_files
from .providers import ProviderUnavailable

JOB_ID=re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')
MAX_CLOUD_POINTS=60000
MAX_DOWNLOAD=512*1024*1024
DEFAULT_THICKNESS=.12
METHOD='roomplan-app API: beam_rec -> wallfield P4 -> TUN3D'
# ScanNet classes of the detector -> procedural furniture kind of this app.
CLASS_KIND={'bed':'bed','sofa':'sofa','chair':'diningChair','table':'diningTable','desk':'desk','cabinet':'cabinet','counter':'cabinet','bookshelf':'wardrobe','refrigerator':'fridge','toilet':'toilet','sink':'sink','bathtub':'bathtub','garbagebin':'custom'}
CLASS_NAME={'bed':'Кровать','sofa':'Диван','chair':'Стул','table':'Стол','desk':'Рабочий стол','cabinet':'Шкаф/тумба','counter':'Столешница','bookshelf':'Стеллаж','refrigerator':'Холодильник','toilet':'Унитаз','sink':'Раковина','bathtub':'Ванна','garbagebin':'Корзина'}
# Doors/windows arrive separately as wall openings; curtains and pictures are not furniture.
NOT_FURNITURE={'door','window','curtain','showercurtrain','picture'}
ROOM_COLORS=['#e9dfca','#e8d9c8','#ded7ca','#e0e5df','#e6dccf','#dfe2d6']
PLY_TYPES={'char':'i1','int8':'i1','uchar':'u1','uint8':'u1','short':'i2','int16':'i2','ushort':'u2','uint16':'u2','int':'i4','int32':'i4','uint':'u4','uint32':'u4','float':'f4','float32':'f4','double':'f8','float64':'f8'}

def config():
    """-> ((url, token, options), None) or (None, reason)."""
    url=os.environ.get('ROOMPLAN_VIDEO_API_URL','').strip().rstrip('/')
    token=os.environ.get('ROOMPLAN_VIDEO_API_TOKEN','').strip()
    if not url or not token: return None,'disabled; set ROOMPLAN_VIDEO_API_URL and ROOMPLAN_VIDEO_API_TOKEN'
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme not in ('http','https') or not parsed.hostname: return None,'ROOMPLAN_VIDEO_API_URL must be an http(s) URL'
    # The token and the video travel with the requests: plain HTTP only to this machine (local service or SSH tunnel).
    if parsed.scheme=='http' and parsed.hostname not in ('127.0.0.1','localhost','::1'): return None,'ROOMPLAN_VIDEO_API_URL must use https for a remote host (plain http only on loopback)'
    try: options=json.loads(os.environ.get('ROOMPLAN_VIDEO_API_OPTIONS','') or '{}')
    except json.JSONDecodeError: options=None
    if not isinstance(options,dict): return None,'ROOMPLAN_VIDEO_API_OPTIONS must be a JSON object'
    return (url,token,options),None

def health():
    cfg,reason=config()
    if cfg: return {'available':True,'mode':'remote-service','method':METHOD,'status':'configured; connectivity checked on upload','externalUpload':True}
    return {'available':False,'mode':'synthetic-demo-stub','method':METHOD,'status':reason,'externalUpload':False}

def _require():
    cfg,reason=config()
    if not cfg: raise ProviderUnavailable(f'Сервис реконструкции по видео не настроен ({reason}). Видео не отправлялось.')
    return cfg

def _call(path,data=None,content_type=None,timeout=60,limit=8*1024*1024):
    url,token,_=_require()
    headers={'X-Token':token,'Accept':'application/json'}
    if content_type: headers['Content-Type']=content_type
    req=urllib.request.Request(url+path,data=data,headers=headers,method='POST' if data is not None else 'GET')
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            raw=response.read(limit+1)
    except urllib.error.HTTPError as e:
        if e.code in (401,403): raise ProviderUnavailable('Сервис реконструкции отклонил токен ROOMPLAN_VIDEO_API_TOKEN.') from e
        if e.code==404: raise ValueError('Задача не найдена в сервисе реконструкции') from e
        if e.code<500:
            try: detail=json.loads(e.read(65536)).get('detail')
            except Exception: detail=None
            raise ValueError(f'Сервис реконструкции отклонил запрос: {detail or e.code}') from e
        raise ProviderUnavailable(f'Сервис реконструкции ответил ошибкой {e.code}. Повторите позже.') from e
    except (urllib.error.URLError,TimeoutError,OSError) as e:
        raise ProviderUnavailable(f'Сервис реконструкции недоступен ({type(e).__name__}). Проверьте ROOMPLAN_VIDEO_API_URL и сеть.') from e
    if len(raw)>limit: raise ProviderUnavailable('Ответ сервиса реконструкции слишком большой')
    return raw

def _json(raw):
    try: value=json.loads(raw)
    except (ValueError,UnicodeDecodeError) as e: raise ProviderUnavailable('Сервис реконструкции вернул не JSON') from e
    if not isinstance(value,dict): raise ProviderUnavailable('Сервис реконструкции вернул неожиданный ответ')
    return value

def public_job(job):
    video=job.get('video') if isinstance(job.get('video'),dict) else {}
    out={k:job.get(k) for k in ('id','status','stage','message','error','created','claimed','finished')}
    # Worker messages may carry whole stage reports (JSON); the dialog needs one line.
    for key,limit in (('stage',100),('message',160),('error',1000)):
        if out[key] is not None: out[key]=' '.join(str(out[key]).split())[:limit]
    out['video']={'name':video.get('name'),'size':video.get('size')}
    return out

def _multipart(fields,filename,mime,data):
    boundary='roomplan-'+uuid.uuid4().hex
    safe=re.sub(r'["\r\n\\]','_',filename)[:200] or 'video.mp4'
    parts=[f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k,v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="{safe}"\r\nContent-Type: {mime}\r\n\r\n'.encode()+data+b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    return b''.join(parts),f'multipart/form-data; boundary={boundary}'

def submit_video(payload):
    files=validate_files(payload.get('files',[]),1)
    if not files: raise ValueError('Добавьте видео помещения')
    f=files[0]
    if not f['mime'].startswith('video/') and not re.search(r'\.(mp4|mov|m4v|webm|mkv|avi|3gp)$',f['name'],re.I): raise ValueError('Ожидается видеофайл (MP4, MOV, WebM)')
    _,_,options=_require()
    body,content_type=_multipart({'include_cloud':'true','formats':'json','options':json.dumps(options)},f['name'],f['mime'] if f['mime'].startswith('video/') else 'application/octet-stream',base64.b64decode(f['data']))
    job=_json(_call('/api/jobs',body,content_type,timeout=900))
    if not isinstance(job.get('id'),str) or not JOB_ID.fullmatch(job['id']): raise ProviderUnavailable('Сервис реконструкции не вернул идентификатор задачи')
    return {'job':public_job(job)}

def poll_video(job_id):
    if not isinstance(job_id,str) or not JOB_ID.fullmatch(job_id) or '..' in job_id: raise ValueError('Неверный идентификатор задачи')
    job=_json(_call('/api/jobs/'+urllib.parse.quote(job_id)))
    out={'job':public_job(job)}
    if job.get('status')!='done': return out
    names=job.get('files') if isinstance(job.get('files'),list) else []
    if 'result.json' not in names: raise ProviderUnavailable('Задача завершена без result.json')
    result=_json(_call(f'/api/jobs/{urllib.parse.quote(job_id)}/files/result.json'))
    cloud=None
    if 'cloud.ply' in names:
        try: cloud=read_ply(_call(f'/api/jobs/{urllib.parse.quote(job_id)}/files/cloud.ply',timeout=600,limit=MAX_DOWNLOAD))
        except ValueError: cloud=None
    video=job.get('video') if isinstance(job.get('video'),dict) else {}
    project,warnings=result_to_project(result,cloud,job_id=job_id,video_name=str(video.get('name') or 'video'))
    if 'cloud.ply' in names and cloud is None: warnings.append('Облако точек не прочитано: слой облака пуст.')
    out.update(project=project,warnings=warnings)
    return out

def read_ply(raw):
    """PLY (binary little/big endian or ascii) -> float32 xyz array. Vertex list properties unsupported."""
    end=raw.find(b'end_header')
    if not raw.startswith(b'ply') or end<0: raise ValueError('Не PLY')
    body=raw[raw.index(b'\n',end)+1:]
    fmt,n,props,element=None,0,[],None
    for line in raw[:end].decode('ascii','replace').splitlines():
        parts=line.split()
        if not parts: continue
        if parts[0]=='format': fmt=parts[1]
        elif parts[0]=='element':
            element=parts[1]
            if element=='vertex': n=int(parts[2])
        elif parts[0]=='property' and element=='vertex':
            if parts[1]=='list' or parts[1] not in PLY_TYPES: raise ValueError('Неподдерживаемое свойство PLY')
            props.append((parts[2],PLY_TYPES[parts[1]]))
    names=[p[0] for p in props]
    if not all(c in names for c in 'xyz'): raise ValueError('В PLY нет координат xyz')
    if fmt=='ascii':
        rows=[r.split() for r in body.decode('ascii','replace').splitlines()[:n]]
        data=np.asarray(rows,dtype=np.float64)
        return data[:,[names.index(c) for c in 'xyz']].astype(np.float32)
    if fmt not in ('binary_little_endian','binary_big_endian'): raise ValueError('Неизвестный формат PLY')
    order='<' if fmt=='binary_little_endian' else '>'
    dtype=np.dtype([(name,order+t) for name,t in props])
    if len(body)<n*dtype.itemsize: raise ValueError('PLY обрезан')
    arr=np.frombuffer(body,dtype=dtype,count=n)
    return np.stack([arr['x'],arr['y'],arr['z']],1).astype(np.float32)

def _num(value,default=0.):
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else default

def _pt(value):
    if not isinstance(value,(list,tuple)) or len(value)<2: raise ValueError('Неверная точка в результате сервиса')
    return _num(value[0],math.nan),_num(value[1],math.nan)

def dominant_angle(segments):
    """Length-weighted Manhattan direction of 2D segments, in (-pi/4, pi/4]."""
    s=c=0.
    for (ax,ay),(bx,by) in segments:
        length=math.hypot(bx-ax,by-ay);angle=math.atan2(by-ay,bx-ax)
        s+=length*math.sin(4*angle);c+=length*math.cos(4*angle)
    return math.atan2(s,c)/4 if s or c else 0.

def floor_ceiling(z):
    """Densest 5 cm layers below/above the median height of the cloud -> (floor, ceiling) or None.

    Wall bottoms/tops of the result are decoder bounds, not surfaces; on real scans they sit
    up to ~25 cm off the floor plane, while furniture stands on the density peak.
    """
    z=z[np.isfinite(z)]
    if len(z)<1000: return None
    mid=float(np.median(z))
    def peak(v):
        if len(v)<200: return None
        edges=np.arange(v.min(),v.max()+.075,.025)
        h,_=np.histogram(v,edges);window=h[:-1]+h[1:]
        return float(edges[int(np.argmax(window))]+.025)
    floor,ceiling=peak(z[z<mid]),peak(z[z>mid])
    if floor is None or ceiling is None or not 1.5<=ceiling-floor<=10: return None
    return floor,ceiling

def result_to_project(result,cloud=None,job_id='',video_name='video'):
    """roomplan-app result.json (+ optional cloud xyz) -> (project, warnings).

    Source frame: metres, Z up, plan in x-y. Project frame: x/z horizontal, y up. The plan is
    rotated so the dominant wall direction is axis-aligned, mirrored y -> -z (keeps handedness
    with y up) and shifted so the wall envelope starts at the origin; floor = median wall bottom.
    """
    if not isinstance(result,dict) or not isinstance(result.get('walls'),list): raise ValueError('Результат сервиса без стен')
    raw_walls=[w for w in result['walls'] if isinstance(w,dict)]
    segments=[(_pt(w.get('a')),_pt(w.get('b'))) for w in raw_walls]
    if not segments or any(not all(map(math.isfinite,a+b)) for a,b in segments): raise ValueError('Сервис не нашёл стен на видео')
    phi=dominant_angle(segments);cos,sin=math.cos(phi),math.sin(phi)
    def plan(x,y): return x*cos+y*sin,-(-x*sin+y*cos)
    corners=[plan(*p) for seg in segments for p in seg]
    ox=min(p[0] for p in corners);oz=min(p[1] for p in corners)
    def to_xz(p):
        x,z=plan(*_pt(p));return [round(x-ox,4),round(z-oz,4)]
    def finite_xz(p):
        try: v=to_xz(p)
        except ValueError: return None
        return v if all(map(math.isfinite,v)) else None
    levels=floor_ceiling(np.asarray(cloud,dtype=np.float64).reshape(-1,3)[:,2]) if cloud is not None and len(cloud) else None
    if levels:
        floor,top=levels;heights=[top-floor]*len(raw_walls);level_source='cloud-density-peaks'
    else:
        floor=float(np.median([_num(w.get('z_floor')) for w in raw_walls]));level_source='walls-median'
        heights=[_num(w.get('z_ceil'),floor+2.7)-_num(w.get('z_floor'),floor) for w in raw_walls]
    heights=[min(10.,max(.1,h)) for h in heights]
    ceiling=round(min(10.,max(1.5,float(np.median(heights)))),2)
    warnings=['План построен сервисом roomplan-app по видео: стены и комнаты — декодер стен, мебель — детектор TUN3D, размеры предметов — по найденным 3D-боксам. Проверьте и при необходимости исправьте модель.',
              f'Высота потолка {ceiling} м оценена по {"плотности облака точек (пол и потолок)" if levels else "стенам результата"}; толщина стен по видео не определяется, принято {round(DEFAULT_THICKNESS*100)} см.']
    stem=video_name.rsplit('.',1)[0][:150] or 'видео'
    p=dict(version=1,name=f'План по видео · {stem}',ceilingHeight=ceiling,walls=[],rooms=[],objects=[],openings=[],pointCloud=[],sources=[],assets=[],
           layers={'architecture':True,'furniture':True,'pointCloud':cloud is not None and len(cloud)>0,'boxes':False,'research':False})
    wall_ids={}
    for i,(w,h) in enumerate(zip(raw_walls,heights)):
        a,b=to_xz(w['a']),to_xz(w['b'])
        if math.dist(a,b)<.01: continue
        wall_ids[i]=f'w{len(p["walls"])+1}'  # openings reference walls by index in result.json
        p['walls'].append(dict(id=f'w{len(p["walls"])+1}',a=a,b=b,height=round(h,3),thickness=DEFAULT_THICKNESS))
    holes=0
    for r in result.get('rooms') or []:
        if not isinstance(r,dict) or not isinstance(r.get('poly'),list): continue
        poly=[finite_xz(v) for v in r['poly']]
        if None in poly: continue
        if len(poly)>3 and poly[0]==poly[-1]: poly.pop()
        if len(poly)<3 or len(poly)>1000: continue
        area=abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1])))/2
        if area<.01: continue
        holes+=len(r.get('holes') or [])
        n=len(p['rooms'])
        p['rooms'].append(dict(id=f'r{n+1}',name=f'Помещение {n+1} · {area:.1f} м²',polygon=poly,color=ROOM_COLORS[n%len(ROOM_COLORS)]))
    if holes: warnings.append(f'Внутренние контуры комнат ({holes}) не перенесены: колонны и выступы проверьте вручную.')
    catalog_by_kind={}
    for c in CATALOG: catalog_by_kind.setdefault(c['kind'],c)
    skipped={}
    for o in result.get('objects') or []:
        if not isinstance(o,dict): continue
        cls=str(o.get('class',''))
        size=o.get('size') if isinstance(o.get('size'),list) and len(o['size'])==3 else None
        centre=finite_xz(o.get('center'))
        if cls in NOT_FURNITURE or cls not in CLASS_KIND or size is None or centre is None:
            skipped[cls]=skipped.get(cls,0)+1;continue
        dx,dy,dz=[min(100.,max(.01,_num(v,.01))) for v in size]
        kind=CLASS_KIND[cls]
        if cls=='table' and dz<.55: kind='coffeeTable'
        entry=catalog_by_kind.get(kind)
        x,z=centre
        bottom=max(0.,_num(o.get('z_bottom'),floor)-floor)
        score=_num(o.get('score'))
        p['objects'].append(dict(id=f'o{len(p["objects"])+1}',catalogId=entry['id'] if entry else 'custom',name=f'{CLASS_NAME.get(cls,cls)} · найдено ({score:.2f})',kind=kind,
            position=[x,round(bottom,4),z],rotation=round(_num(o.get('yaw'))-phi,5),size=[round(dx,4),round(dz,4),round(dy,4)],
            color=entry['color'] if entry else '#a7aca0',mount='floor',detection={'class':cls,'score':round(score,4),'source':'TUN3D'}))
    if skipped: warnings.append('Не перенесены как мебель: '+', '.join(f'{k or "без класса"} × {v}' for k,v in sorted(skipped.items()))+' (двери и окна добавлены проёмами в стенах).')
    dropped=0
    for kind,items in (('door',result.get('doors') or []),('window',result.get('windows') or [])):
        for it in items:
            if not isinstance(it,dict) or not isinstance(it.get('corners'),list) or len(it['corners'])<2: dropped+=1;continue
            wid=wall_ids.get(it.get('wall'))
            wall=next((w for w in p['walls'] if w['id']==wid),None)
            if wall is None: dropped+=1;continue
            c=it['corners']  # 2 bottom then 2 top corners, world z; 'sill' is relative to the decoder wall bottom
            if len(c)>=3 and all(isinstance(v,list) and len(v)>=3 for v in c[:3]):
                sill=max(0.,_num(c[0][2],floor)-floor);top=min(wall['height'],_num(c[2][2],floor)-floor)
            else: sill=max(0.,_num(it.get('sill')));top=min(wall['height'],sill+max(0.,_num(it.get('height'))))
            a,b=finite_xz(it['corners'][0]),finite_xz(it['corners'][1])
            if a is None or b is None or top-sill<.1 or math.dist(a,b)<.05: dropped+=1;continue
            p['openings'].append(dict(id=f'{kind}{len(p["openings"])+1}',kind=kind,wallId=wid,a=a,b=b,height=round(top-sill,3),sillHeight=round(sill,3),detection={'score':round(_num(it.get('score')),4),'source':str(it.get('src',''))[:50]}))
    if dropped: warnings.append(f'Проёмы вне стен или вне высоты стены пропущены: {dropped}.')
    if cloud is not None and len(cloud):
        xyz=np.asarray(cloud,dtype=np.float64).reshape(-1,3)
        xyz=xyz[np.isfinite(xyz).all(1)]
        if len(xyz)>MAX_CLOUD_POINTS: xyz=xyz[np.sort(np.random.default_rng(0).choice(len(xyz),MAX_CLOUD_POINTS,replace=False))]
        px=xyz[:,0]*cos+xyz[:,1]*sin-ox;pz=-(-xyz[:,0]*sin+xyz[:,1]*cos)-oz;py=xyz[:,2]-floor
        keep=(np.abs(px)<999)&(np.abs(pz)<999)&(np.abs(py)<999)
        p['pointCloud']=np.round(np.stack([px,py,pz],1)[keep],3).tolist()
    meta=result.get('meta') if isinstance(result.get('meta'),dict) else {}
    quality=meta.get('quality') if isinstance(meta.get('quality'),dict) else {}
    red=[str(g) for g in quality.get('red_gates') or []]
    if red: warnings.append('Сервис отметил слабые места реконструкции ('+', '.join(red)+'): геометрию стоит проверить по облаку точек.')
    pipeline=meta.get('pipeline') if isinstance(meta.get('pipeline'),dict) else {}
    scale=meta.get('scale') if isinstance(meta.get('scale'),dict) else {}
    p['provenance']={'mode':'video','method':METHOD,'isSynthetic':False,'inputUsedForGeometry':True,
        'service':{'job':job_id,'areaM2':meta.get('area_m2'),'wallsDecoder':meta.get('walls_decoder'),'pipeline':pipeline.get('selected') or pipeline.get('pipeline'),'quality':quality.get('status'),'redGates':red,'scaleSource':scale.get('source'),'frames':meta.get('frames'),'points':meta.get('points')},
        'alignment':{'sourceFrame':'metres, Z up, plan x-y (service result)','rotationRad':round(phi,6),'floorZ':round(floor,4),'floorSource':level_source,'offset':[round(ox,4),round(oz,4)],'mapping':'x=(R(-rotation)*[x,y])_x-offset0; z=-(R(-rotation)*[x,y])_y-offset1; y=z-floorZ'}}
    p['assets'].append({'id':str(uuid.uuid4()),'name':'roomplan-result.json','mime':'application/json','data':base64.b64encode(json.dumps(result,ensure_ascii=False).encode()).decode()})
    return p,warnings
