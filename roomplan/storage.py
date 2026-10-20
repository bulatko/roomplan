import base64
import copy
import io
import json
import re
import zipfile
from . import scenes
from .model import MAX_TOTAL, validate_project

def save_project(project):
    p,_=validate_project(project);p=copy.deepcopy(p);out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for kind in ['sources','assets']:
            for index,f in enumerate(p[kind]):
                safe=re.sub(r'[^\w.\-]','_',f['name'])[:150] or 'file'
                path=f'{kind}/{index:03d}-{safe}'
                z.writestr(path,base64.b64decode(f.pop('data')));f['archivePath']=path
        for s in p.get('scenes',[]):  # bundles live on the server; the portable project carries them along
            bundle=scenes.bundle_path(s['id'])
            if bundle.is_file():
                path=f'scenes/{s["id"]}.tar.gz';z.write(bundle,path,compress_type=zipfile.ZIP_STORED);s['archivePath']=path
        z.writestr('project.json',json.dumps(p,ensure_ascii=False,allow_nan=False,indent=2))
        z.writestr('README.txt','RoomPlan portable project v1. project.json contains metric geometry and paths to every original source and generated asset. Files are never executable. Open the ZIP in RoomPlan to continue editing.\n')
    return out.getvalue()

def load_project(data):
    try:
        raw=base64.b64decode(data,validate=True)
        if len(raw)>MAX_TOTAL: raise ValueError('Архив слишком большой')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            infos=z.infolist()
            if len(infos)>205 or sum(i.file_size for i in infos)>MAX_TOTAL: raise ValueError('Архив превышает лимиты')
            if len({i.filename for i in infos})!=len(infos): raise ValueError('Повторяющиеся пути в архиве')
            if any(i.filename.startswith(('/','\\')) or '..' in i.filename.replace('\\','/').split('/') for i in infos): raise ValueError('Небезопасный путь в архиве')
            if z.getinfo('project.json').file_size>30*1024*1024: raise ValueError('project.json слишком большой')
            p=json.loads(z.read('project.json'))
            for kind in ['sources','assets']:
                for f in p.get(kind,[]):
                    path=f.pop('archivePath',None)
                    if path:
                        if not path.startswith(kind+'/'): raise ValueError('Неверная ссылка на ресурс')
                        f['data']=base64.b64encode(z.read(path)).decode()
            scenes.validate_scene_refs(p.get('scenes',[]))
            for s in p.get('scenes',[]):
                path=s.pop('archivePath',None)
                if path:
                    if path!=f'scenes/{s["id"]}.tar.gz': raise ValueError('Неверная ссылка на пакет сцены')
                    s.update(scenes.store_bundle(s['id'],z.read(path),source=s.get('source')))
        return validate_project(p)
    except ValueError: raise
    except Exception as e: raise ValueError('Не удалось открыть переносимый проект RoomPlan') from e
