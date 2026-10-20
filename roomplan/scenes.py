"""Shared scene space: reconstruction bundles kept on the server for the next functions (detection, ...).

A scene bundle (format roomplan-scene/1, produced by the roomplan-app reconstruction service) is a
tar.gz with one top directory: images/ (posed RGB frames), poses.json (camera, T_cw per frame,
self-check), cloud.ply, plan.json, manifest.json — all in one world (metres, Z up, floor z=0).
The project keeps only a reference with a summary (`project.scenes`); the bundle itself lives in
ROOMPLAN_DATA_DIR/scenes/<id>/ (default ./data) and travels inside the portable project ZIP.
"""
import hashlib
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path

SCENE_ID=re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')
MAX_BUNDLE=1024*1024*1024
MAX_MEMBERS=20000
FORMAT='roomplan-scene/1'

def data_dir():
    return Path(os.environ.get('ROOMPLAN_DATA_DIR') or Path(__file__).resolve().parent.parent/'data')

def scene_dir(sid):
    if not isinstance(sid,str) or not SCENE_ID.fullmatch(sid) or '..' in sid: raise ValueError('Неверный идентификатор сцены')
    return data_dir()/'scenes'/sid

def bundle_path(sid):
    return scene_dir(sid)/'scene.tar.gz'

def summary(sid):
    path=scene_dir(sid)/'scene.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() and bundle_path(sid).is_file() else None

def inspect_bundle(raw):
    """Check a bundle without extracting it -> facts from poses.json. Rejects unsafe or foreign archives."""
    if len(raw)>MAX_BUNDLE: raise ValueError('Пакет сцены больше 1 ГБ')
    try: tf=tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')
    except Exception as e: raise ValueError('Пакет сцены не является tar.gz') from e
    with tf:
        members=tf.getmembers()
        if not members or len(members)>MAX_MEMBERS: raise ValueError('Пакет сцены пуст или слишком велик')
        top=members[0].name.split('/')[0]
        for m in members:
            parts=m.name.split('/')
            if m.name.startswith(('/','\\')) or '..' in parts or parts[0]!=top: raise ValueError('Небезопасный путь в пакете сцены')
            if not (m.isfile() or m.isdir()): raise ValueError('Ссылки и спецфайлы в пакете сцены не допускаются')
        names={m.name for m in members}
        for need in ('poses.json','cloud.ply'):
            if f'{top}/{need}' not in names: raise ValueError(f'В пакете сцены нет {need}')
        poses=json.loads(tf.extractfile(f'{top}/poses.json').read())
        if poses.get('format')!=FORMAT: raise ValueError(f'Неизвестный формат пакета сцены: {poses.get("format")!r}')
        frames=poses.get('frames') if isinstance(poses.get('frames'),list) else []
        images=sum(1 for n in names if n.startswith(f'{top}/images/') and n.count('/')==2)
        checks=poses.get('checks') if isinstance(poses.get('checks'),dict) else {}
        counts=poses.get('counts') if isinstance(poses.get('counts'),dict) else {}
        return {'format':FORMAT,'top':top,'frames':len(frames),'images':images,'cloudPoints':counts.get('cloud_points'),
                'hasPlan':f'{top}/plan.json' in names,'checksOk':checks.get('ok'),'world':(poses.get('world') or {}).get('note')}

def store_bundle(sid,raw,source=None):
    """Validate and keep a bundle under its id (atomic replace) -> summary for project.scenes."""
    facts=inspect_bundle(raw)
    d=scene_dir(sid);d.mkdir(parents=True,exist_ok=True)
    tmp=d/'scene.tar.gz.part';tmp.write_bytes(raw);tmp.replace(d/'scene.tar.gz')
    info={'id':sid,**facts,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'storedAt':round(time.time(),3),
          'source':source if isinstance(source,dict) else {}}
    (d/'scene.json').write_text(json.dumps(info,ensure_ascii=False,indent=1),encoding='utf-8')
    return info

def validate_scene_refs(scenes):
    if not isinstance(scenes,list) or len(scenes)>20: raise ValueError('scenes: ожидается не более 20 ссылок на сцены')
    ids=set()
    for s in scenes:
        if not isinstance(s,dict) or not isinstance(s.get('id'),str) or not SCENE_ID.fullmatch(s['id']) or s['id'] in ids: raise ValueError('scenes: нужен уникальный id сцены')
        ids.add(s['id'])
