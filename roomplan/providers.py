"""Explicit local single-image mesh adapter. No fake geometry fallback."""
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
import shutil
from .model import validate_files
from .glb import validate_glb

class ProviderUnavailable(Exception): pass

def health():
    from .video_service import health as video_health  # imports ProviderUnavailable from here
    from .functions import health as detection_health
    url=os.environ.get('ROOMPLAN_RECONSTRUCTION_URL','')
    return {'status':'ok','version':1,'videoExport':{'available':bool(shutil.which('ffmpeg')),'method':'fixed-frame FFmpeg VP9','fps':15},'providers':{'plan':{'available':True,'method':'CPU threshold + axis-line detection'},'video':video_health(),'detection':detection_health(),'photos':{'available':False,'mode':'synthetic-demo-stub'},'object':{'configured':bool(url),'available':None if url else False,'method':'local Stable Fast 3D adapter','status':'configured; connectivity checked on inference' if url else 'disabled; set ROOMPLAN_RECONSTRUCTION_URL','externalUpload':False}}}

def reconstruct_object(payload):
    files=validate_files(payload.get('files',[]),20)
    if not files: raise ValueError('Добавьте фотографию объекта')
    url=os.environ.get('ROOMPLAN_RECONSTRUCTION_URL','')
    if not url: raise ProviderUnavailable('Локальная реконструкция SF3D не настроена. Запустите GPU-сервис и задайте ROOMPLAN_RECONSTRUCTION_URL. Нейросетевая модель не запускалась.')
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='http' or parsed.hostname not in ['127.0.0.1','localhost','::1']: raise ProviderUnavailable('Адаптер принимает только локальный HTTP-сервис (loopback). Настройте локальное перенаправление порта GPU-сервера.')
    req=urllib.request.Request(url,data=json.dumps({'files':[files[0]],'name':str(payload.get('name','Объект SF3D'))[:200]}).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=600) as response:
            raw=response.read(101*1024*1024)
        if len(raw)>100*1024*1024: raise ValueError('Ответ модели слишком большой')
        result=json.loads(raw)
        asset=result['asset'];validate_files([asset])
        if asset['mime']!='model/gltf-binary': raise ValueError('Сервис не вернул GLB')
        validate_glb(base64.b64decode(asset['data'],validate=True))
    except Exception as e: raise ProviderUnavailable(f'Локальный сервис не завершил реконструкцию ({type(e).__name__}). Проверьте GPU-сервис; результат не был подменён.') from e
    aid=str(uuid.uuid4());asset['id']=aid;asset['method']='Stable Fast 3D';asset['isSyntheticProxy']=False
    obj={'id':str(uuid.uuid4()),'catalogId':'custom','kind':'custom','name':str(payload.get('name','Объект SF3D'))[:200],'position':[2,0,2],'size':[1,1,1],'rotation':0,'mount':'floor','color':'#b9c4b3','assetId':aid}
    return {'asset':asset,'object':obj,'sources':[dict(f,id=str(uuid.uuid4())) for f in files],'warnings':['SF3D использует первое изображение. Физические размеры неизвестны: задайте их вручную. Скрытые поверхности являются оценкой модели.']}
