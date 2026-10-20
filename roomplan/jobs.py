"""Job journal of the hub. Every function job the app starts (video reconstruction now, detection later)
is recorded on the server together with its input, so the job stays visible after a page reload and
its result can be opened later, with the original video back in project.sources.
Layout: ROOMPLAN_DATA_DIR/jobs/<jobId>/{job.json,input.<ext>}.
"""
import base64
import json
import re
import time
from .scenes import data_dir

JOB_ID=re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')

def job_dir(jid):
    if not isinstance(jid,str) or not JOB_ID.fullmatch(jid) or '..' in jid: raise ValueError('Неверный идентификатор задачи')
    return data_dir()/'jobs'/jid

def record(jid,kind,source,**extra):
    """Keep the input file and what was asked; source = {name, mime, data(base64)} as in the API."""
    d=job_dir(jid);d.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(source['data'])
    m=re.search(r'\.([A-Za-z0-9]{1,8})$',source.get('name',''));ext='.'+m.group(1).lower() if m else '.bin'
    (d/f'input{ext}').write_bytes(raw)
    meta={'id':jid,'kind':kind,'createdAt':round(time.time(),3),'input':{'name':str(source.get('name',''))[:300],'mime':str(source.get('mime',''))[:100],'bytes':len(raw),'file':f'input{ext}'},**extra}
    (d/'job.json').write_text(json.dumps(meta,ensure_ascii=False,indent=1),encoding='utf-8')
    return meta

def get(jid):
    try: path=job_dir(jid)/'job.json'
    except ValueError: return None
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

def recorded():
    root=data_dir()/'jobs'
    metas=[json.loads(p.read_text(encoding='utf-8')) for p in root.glob('*/job.json')] if root.is_dir() else []
    return sorted(metas,key=lambda m:m.get('createdAt',0),reverse=True)

def input_source(jid):
    """The recorded input as a project source (same shape as uploaded files), or None."""
    meta=get(jid)
    if not meta: return None
    path=job_dir(jid)/meta['input']['file']
    if not path.is_file(): return None
    return {'id':f'input-{jid}','name':meta['input']['name'],'mime':meta['input']['mime'],'data':base64.b64encode(path.read_bytes()).decode()}
