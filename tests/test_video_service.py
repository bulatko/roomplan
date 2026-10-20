import base64
import json
import math
import os
import re
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import numpy as np
from roomplan import jobs, scenes
from roomplan.model import validate_project
try: from test_scenes import make_bundle            # unittest discover -s tests
except ImportError: from tests.test_scenes import make_bundle
from roomplan.providers import health
from roomplan.video_service import DEFAULT_THICKNESS, dominant_angle, floor_ceiling, public_job, read_ply, result_to_project
from server import Handler, load_env

PHI=math.radians(30)  # plan rotation of the synthetic scan
FLOOR=-.2;TOP=2.6     # decoder wall bounds (not the true floor)

def rot(x,y): return [x*math.cos(PHI)-y*math.sin(PHI),x*math.sin(PHI)+y*math.cos(PHI)]

def synthetic_result():
    """4 x 3 m room rotated by 30 deg, shifted; Z up like the service world."""
    corners=[rot(x,y) for x,y in [(0,0),(4,0),(4,3),(0,3)]]
    corners=[[x+5,y-2] for x,y in corners]
    walls=[dict(id=i,a=corners[i],b=corners[(i+1)%4],z_floor=FLOOR,z_ceil=TOP,length_m=4 if i%2==0 else 3) for i in range(4)]
    centre=[a+b for a,b in zip(rot(1.5,1),[5,-2])]
    # Door on wall 0 (from 1 m to 1.9 m along it), bottom at floor, 2.1 m high; window above the wall top.
    door=[rot(1,0),rot(1.9,0)];door=[[x+5,y-2] for x,y in door]
    window=[rot(2,3),rot(3,3)];window=[[x+5,y-2] for x,y in window]
    return {'meta':{'area_m2':12.0,'walls_decoder':'test','quality':{'status':'warn','red_gates':['fuse']},'scale':{'source':'da3'}},
            'rooms':[{'id':0,'poly':corners,'holes':[],'area_m2':12.0,'z_floor':FLOOR,'z_ceil':TOP}],'walls':walls,
            'objects':[{'id':0,'class':'bed','score':.9,'center':centre+[.4],'z_bottom':.1,'size':[2.0,1.6,.5],'yaw':PHI+.1},
                       {'id':1,'class':'table','score':.7,'center':centre+[.2],'z_bottom':.1,'size':[1.0,.6,.4],'yaw':PHI},
                       {'id':2,'class':'door','score':.6,'center':centre+[1],'z_bottom':.1,'size':[.9,.1,2.1],'yaw':0},
                       {'id':3,'class':'picture','score':.5,'center':centre+[1.5],'z_bottom':1.2,'size':[.5,.05,.4],'yaw':0}],
            'doors':[{'id':0,'wall':0,'u':1,'width':.9,'height':2.3,'sill':.3,'kind':'door','corners':[door[0]+[.1],door[1]+[.1],door[1]+[2.2],door[0]+[2.2]],'score':.5,'src':'t'}],
            'windows':[{'id':1,'wall':2,'u':1,'width':1,'height':1,'sill':3,'kind':'window','corners':[window[0]+[2.8],window[1]+[2.8],window[1]+[3.8],window[0]+[3.8]],'score':.4,'src':'t'}]}

def synthetic_cloud():
    """Dense floor at z=0.1 and ceiling at z=2.6 over the room, sparse wall points and a table top."""
    rng=np.random.default_rng(3);pts=[]
    for z,n in [(.1,20000),(2.6,15000),(.7,3000)]:
        uv=rng.uniform([0,0],[4,3],(n,2));pts.append(np.c_[[rot(u,v) for u,v in uv],np.full(n,z)+rng.normal(0,.005,n)])
    uv=np.c_[rng.uniform(0,4,4000),np.zeros(4000)];pts.append(np.c_[[rot(u,v) for u,v in uv],rng.uniform(.1,2.6,4000)])
    xyz=np.concatenate(pts);xyz[:,0]+=5;xyz[:,1]-=2
    return xyz.astype(np.float32)

def two_rooms(doorway=True):
    """Rooms A x 0..4 and B x 4.16..8 (z 0..3, service frame) split by a 16 cm partition;
    optional 0.9 m doorway at y 1.2..2.1 with a lintel at 2.0 m. Floor z=0, ceiling 2.6."""
    def contour(x0,x1): return [[x0,0],[x1,0],[x1,3],[x0,3]]
    rooms=[contour(0,4),contour(4.16,8)]
    walls=[dict(id=len(rooms[0])*k+i,a=r[i],b=r[(i+1)%4],z_floor=-.2,z_ceil=2.7) for k,r in enumerate(rooms) for i in range(4)]
    result={'meta':{},'rooms':[{'id':k,'poly':r,'holes':[],'z_floor':-.2,'z_ceil':2.7} for k,r in enumerate(rooms)],'walls':walls,'objects':[],'doors':[],'windows':[]}
    rng=np.random.default_rng(5);pts=[]
    for x0,x1 in ((0,4),(4.16,8)):
        for z,n in ((0.,20000),(2.6,15000)): pts.append(np.c_[rng.uniform(x0,x1,n),rng.uniform(0,3,n),np.full(n,z)])
    for x in (4.,4.16):  # both partition faces
        f=np.c_[np.full(12000,x),rng.uniform(0,3,12000),rng.uniform(.05,2.6,12000)]
        if doorway: f=f[~((f[:,1]>1.2)&(f[:,1]<2.1)&(f[:,2]<2.0))]
        pts.append(f)
    if doorway: pts.append(np.c_[rng.uniform(4.,4.16,400),rng.uniform(1.2,2.1,400),np.zeros(400)])  # threshold floor seen through
    return result,np.concatenate(pts).astype(np.float32)

def ply_binary(xyz):
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode()
    return header+b''.join(struct.pack('<fffBBB',*map(float,p),10,20,30) for p in xyz)

class ConverterTests(unittest.TestCase):
    def test_plan_is_axis_aligned_metric_and_valid(self):
        p,w=result_to_project(synthetic_result(),job_id='j1',video_name='room.mp4')
        p,v=validate_project(p)
        self.assertEqual(len(p['walls']),4);self.assertEqual(len(p['rooms']),1)
        for wall in p['walls']: self.assertTrue(abs(wall['a'][0]-wall['b'][0])<1e-3 or abs(wall['a'][1]-wall['b'][1])<1e-3)
        xs=[c[0] for wl in p['walls'] for c in (wl['a'],wl['b'])];zs=[c[1] for wl in p['walls'] for c in (wl['a'],wl['b'])]
        self.assertAlmostEqual(min(xs),0,places=3);self.assertAlmostEqual(min(zs),0,places=3)
        self.assertEqual(sorted([round(max(xs)-min(xs),3),round(max(zs)-min(zs),3)]),[3.0,4.0])
        self.assertAlmostEqual(p['ceilingHeight'],2.8,places=2)  # no cloud: decoder bounds
        self.assertEqual(p['provenance']['alignment']['floorSource'],'walls-median')
        self.assertFalse(p['provenance']['isSynthetic']);self.assertTrue(p['provenance']['inputUsedForGeometry'])
        self.assertEqual(p['provenance']['service']['redGates'],['fuse'])
        self.assertEqual(json.loads(base64.b64decode(p['assets'][0]['data']))['meta']['area_m2'],12.0)
        self.assertTrue(any('fuse' in x for x in w));self.assertTrue(any('door × 1' in x and 'picture × 1' in x for x in w))
        self.assertAlmostEqual(dominant_angle([((0,0),(math.cos(.3),math.sin(.3)))]),.3,places=6)

    def test_objects_keep_size_and_heading(self):
        result=synthetic_result();p,_=result_to_project(result)
        bed=next(o for o in p['objects'] if o['kind']=='bed');table=next(o for o in p['objects'] if o['detection']['class']=='table')
        self.assertEqual(len(p['objects']),2);self.assertEqual(table['kind'],'coffeeTable')  # 0.4 m high table
        self.assertEqual(bed['size'],[2.0,.5,1.6]);self.assertEqual(bed['mount'],'floor')
        # Heading: the bed's local x axis must point where the service yaw points, after the same plan transform.
        src=result['objects'][0];yaw=src['yaw'];phi=p['provenance']['alignment']['rotationRad'];ox,oz=p['provenance']['alignment']['offset']
        def to_xz(x,y): return x*math.cos(phi)+y*math.sin(phi)-ox,-(-x*math.sin(phi)+y*math.cos(phi))-oz
        c=to_xz(*src['center'][:2]);tip=to_xz(src['center'][0]+math.cos(yaw),src['center'][1]+math.sin(yaw))
        theta=bed['rotation'];self.assertAlmostEqual(tip[0]-c[0],math.cos(theta),places=4);self.assertAlmostEqual(tip[1]-c[1],-math.sin(theta),places=4)  # three.js rotation.y
        self.assertAlmostEqual(bed['position'][0],c[0],places=3);self.assertAlmostEqual(bed['position'][2],c[1],places=3)

    def test_cloud_sets_floor_ceiling_and_openings(self):
        p,w=result_to_project(synthetic_result(),synthetic_cloud())
        p,_=validate_project(p)
        self.assertEqual(p['provenance']['alignment']['floorSource'],'cloud-density-peaks')
        self.assertAlmostEqual(p['provenance']['alignment']['floorZ'],.1,delta=.03)
        self.assertAlmostEqual(p['ceilingHeight'],2.5,delta=.03)
        self.assertTrue(all(abs(o['position'][1])<.02 for o in p['objects']))  # standing on the floor plane
        self.assertEqual(len(p['openings']),1)  # window above the ceiling is dropped
        door=p['openings'][0];self.assertEqual((door['kind'],door['wallId']),('door','w1'))
        self.assertAlmostEqual(door['sillHeight'],0,delta=.03);self.assertAlmostEqual(door['height'],2.1,delta=.03)
        self.assertAlmostEqual(math.dist(door['a'],door['b']),.9,places=3)
        self.assertFalse(p['layers']['pointCloud']);self.assertTrue(p['pointCloud']);self.assertLessEqual(len(p['pointCloud']),60000)  # kept, hidden by default
        cloud=np.array(p['pointCloud']);self.assertAlmostEqual(float(np.median(cloud[cloud[:,1]<1,1])),0,delta=.02)
        self.assertTrue(any('пропущены: 1' in x for x in w))

    def test_partition_faces_merge_and_doorway_from_cloud(self):
        result,cloud=two_rooms()
        p,w=result_to_project(result,cloud);p,_=validate_project(p)
        parts=[x for x in p['walls'] if x.get('partition')];self.assertEqual(len(parts),1);part=parts[0]
        self.assertEqual(len(p['walls']),7)  # 8 contour walls, two faces became one partition
        self.assertAlmostEqual(part['thickness'],.16,places=3)
        self.assertAlmostEqual(part['a'][0],4.08,places=3);self.assertAlmostEqual(part['b'][0],4.08,places=3)  # mid-line
        self.assertAlmostEqual(abs(part['a'][1]-part['b'][1]),3+DEFAULT_THICKNESS,places=3)  # reaches into the outer walls
        door=[o for o in p['openings'] if o['wallId']==part['id']];self.assertEqual(len(door),1);door=door[0]
        # Service y 1.2..2.1 -> project z = 3 - y = 0.9..1.8.
        self.assertEqual((door['kind'],door['sillHeight']),('door',0.))
        self.assertAlmostEqual(min(door['a'][1],door['b'][1]),.9,delta=.06);self.assertAlmostEqual(max(door['a'][1],door['b'][1]),1.8,delta=.06)
        self.assertAlmostEqual(door['height'],2.0,delta=.05)
        self.assertTrue(any('Перегородки между помещениями: 1' in x and '16 см' in x and 'по облаку точек: 1' in x for x in w))
        self.assertIn('Все соседние помещения связаны проходами.',w)

    def test_solid_partition_or_no_cloud_gives_no_passage(self):
        result,cloud=two_rooms(doorway=False)
        for c in (cloud,None):
            p,w=result_to_project(result,c);validate_project(p)
            self.assertEqual(sum(1 for x in p['walls'] if x.get('partition')),1);self.assertEqual(p['openings'],[])
            self.assertTrue(any(x.startswith('Без прохода к соседним помещениям: Помещение 1, Помещение 2') for x in w))
        p,_=result_to_project(synthetic_result());self.assertFalse(any(x.get('partition') for x in p['walls']))  # one room: nothing to merge

    def test_detector_door_wins_over_cloud_passage(self):
        result,cloud=two_rooms()
        # The service found the same doorway as a door on room A's face (x=4, y 1.2..2.1).
        result['doors']=[{'id':0,'wall':1,'kind':'door','corners':[[4,1.2,0],[4,2.1,0],[4,2.1,2.0],[4,1.2,2.0]],'score':.6,'src':'t'}]
        p,w=result_to_project(result,cloud);validate_project(p)
        part=next(x for x in p['walls'] if x.get('partition'))
        self.assertEqual([(o['id'],o['wallId']) for o in p['openings']],[('door1',part['id'])])  # no duplicate passage
        self.assertIn('Все соседние помещения связаны проходами.',w)

    def test_one_face_bordering_two_rooms_gives_two_partitions(self):
        # A (0..4 x 0..4) borders B (y 0..2) and C (y 2.16..4) along its face x=4; B|C share y≈2.08.
        rects=[(0,4,0,4),(4.16,8,0,2),(4.16,8,2.16,4)]
        polys=[[[x0,y0],[x1,y0],[x1,y1],[x0,y1]] for x0,x1,y0,y1 in rects]
        walls=[dict(id=4*k+i,a=r[i],b=r[(i+1)%4],z_floor=0,z_ceil=2.7) for k,r in enumerate(polys) for i in range(4)]
        p,w=result_to_project({'meta':{},'rooms':[{'poly':r} for r in polys],'walls':walls,'objects':[],'doors':[],'windows':[]})
        validate_project(p)
        parts=[x for x in p['walls'] if x.get('partition')];self.assertEqual(len(parts),3)
        lengths=sorted(round(math.dist(x['a'],x['b'])-DEFAULT_THICKNESS,1) for x in parts)  # ends reach into the neighbours
        self.assertEqual(lengths,[1.8,2.0,3.8])  # A|B, A|C (each <= 2 m) and B|C (3.84 m)
        self.assertTrue(any('Без прохода к соседним помещениям: Помещение 1, Помещение 2, Помещение 3' in x for x in w))

    def test_kept_result_has_no_machine_paths(self):
        result=synthetic_result();result['meta'].update(detector={'config':'/mnt/vol/ckpt/run01/cfg.py','ckpt':'/mnt/vol/ckpt/run01/epoch_12.pth'},pipeline={'work':'/mnt/vol/jobs/J/work/'})
        p,_=result_to_project(result)
        kept=base64.b64decode(p['assets'][0]['data']).decode()
        self.assertNotIn('/mnt/',kept);self.assertIn('"epoch_12.pth"',kept);self.assertIn('"work"',kept)
        self.assertNotIn('/mnt/',json.dumps(p['provenance']))

    def test_rejects_results_without_walls(self):
        for bad in [{},{'walls':[]},{'walls':[{'a':[0,'x'],'b':[1,1]}]}]:
            with self.assertRaises(ValueError): result_to_project(bad)
        self.assertIsNone(floor_ceiling(np.zeros(10)))

    def test_job_status_is_one_short_line(self):
        job=public_job({'id':'j','status':'running','stage':'beam_rec','message':'STAGE frames 13s '+'{"idx": 1}\n'*500,'log_tail':'x','error':None})
        self.assertLessEqual(len(job['message']),160);self.assertNotIn('\n',job['message']);self.assertNotIn('log_tail',job)

    def test_ply_reader_binary_and_ascii(self):
        xyz=np.array([[1,2,3],[4,5,6.5]],np.float32)
        np.testing.assert_allclose(read_ply(ply_binary(xyz)),xyz)
        ascii_ply=b'ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n4 5 6.5\n'
        np.testing.assert_allclose(read_ply(ascii_ply),xyz)
        for bad in [b'not ply',ply_binary(xyz)[:-5]]:
            with self.assertRaises(ValueError): read_ply(bad)

class FakeService(BaseHTTPRequestHandler):
    """Minimal roomplan-app API: queued -> running -> done (or failed)."""
    token='secret';jobs={};uploads=[];scene_downloads=0
    def log_message(self,*args): pass
    def send(self,status,body,mime='application/json'):
        data=body if isinstance(body,bytes) else json.dumps(body).encode()
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def authorised(self):
        if self.headers.get('X-Token')!=self.token: self.send(401,{'detail':'bad token'});return False
        return True
    def do_POST(self):
        if not self.authorised(): return
        body=self.rfile.read(int(self.headers['Content-Length']))
        name=re.search(rb'filename="([^"]*)"',body).group(1).decode()
        FakeService.uploads.append({'name':name,'body':body,'type':self.headers['Content-Type']})
        jid='20261020-000000-'+('fail00' if 'fail' in name else f'{len(FakeService.jobs):06x}')
        FakeService.jobs[jid]={'id':jid,'status':'queued','stage':None,'message':None,'error':None,'files':[],'video':{'name':name,'size':len(body)},'log_tail':'secret log','polls':0}
        self.send(200,{k:v for k,v in FakeService.jobs[jid].items() if k!='polls'})
    def do_GET(self):
        if not self.authorised(): return
        if self.path=='/api/jobs': return self.send(200,[{k:v for k,v in j.items() if k not in ('polls','log_tail')} for j in FakeService.jobs.values()])
        m=re.fullmatch(r'/api/jobs/([^/]+)(?:/files/([^/]+))?',self.path)
        job=FakeService.jobs.get(m.group(1)) if m else None
        if not job: return self.send(404,{'detail':'no such job'})
        if m.group(2):
            if m.group(2)=='result.json': return self.send(200,synthetic_result())
            if m.group(2)=='cloud.ply': return self.send(200,ply_binary(synthetic_cloud()[::4]),'application/octet-stream')
            if m.group(2)=='scene.tar.gz':
                FakeService.scene_downloads+=1;return self.send(200,make_bundle(top=f'scene_{job["id"]}'),'application/gzip')
            return self.send(404,{'detail':'no such file'})
        job['polls']+=1
        if job['id'].endswith('fail00'): job.update(status='failed',stage='beam_rec',error='poses: too few frames')
        elif job['polls']==1: job.update(status='running',stage='beam_rec',message='VGGT')
        else: job.update(status='done',stage='done',files=['bev.png','cloud.ply','result.json','scene.tar.gz'])
        self.send(200,{k:v for k,v in job.items() if k!='polls'})

def request(url,payload=None):
    req=urllib.request.Request(url,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=60) as r: return r.status,json.load(r)
    except urllib.error.HTTPError as e: return e.code,json.load(e)

class VideoAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servers=[]
        for handler in (FakeService,Handler):
            server=ThreadingHTTPServer(('127.0.0.1',0),handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();cls.servers.append((server,thread))
        cls.service=f'http://127.0.0.1:{cls.servers[0][0].server_port}';cls.url=f'http://127.0.0.1:{cls.servers[1][0].server_port}'
        cls.data=tempfile.TemporaryDirectory()
    @classmethod
    def tearDownClass(cls):
        for server,thread in cls.servers: server.shutdown();server.server_close();thread.join()
        cls.data.cleanup()
    def env(self,**kw):
        values={'ROOMPLAN_VIDEO_API_URL':self.service,'ROOMPLAN_VIDEO_API_TOKEN':'secret','ROOMPLAN_VIDEO_API_OPTIONS':'{"beam_rec":{"target_frames":200}}',
                'ROOMPLAN_DATA_DIR':self.data.name};values.update(kw)
        return patch.dict(os.environ,{k:v for k,v in values.items() if v is not None},clear=False)
    def video(self,name='room.mp4'): return {'files':[{'name':name,'mime':'video/mp4','data':base64.b64encode(b'\x00\x00\x00\x18ftypmp42video-bytes').decode()}]}

    def test_upload_poll_and_convert(self):
        with self.env():
            self.assertTrue(health()['providers']['video']['available']);downloads0=FakeService.scene_downloads
            status,data=request(self.url+'/api/reconstruct/video',self.video('Комната 1.mp4'))
            self.assertEqual(status,200);job=data['job'];self.assertEqual(job['status'],'queued');self.assertNotIn('log_tail',job)
            upload=FakeService.uploads[-1];self.assertEqual(upload['name'],'Комната 1.mp4');self.assertIn(b'ftypmp42video-bytes',upload['body'])
            self.assertIn(b'name="include_cloud"\r\n\r\ntrue',upload['body']);self.assertIn(b'"target_frames": 200',upload['body'])
            status,data=request(self.url+'/api/reconstruct/video/'+job['id'])
            self.assertEqual((status,data['job']['status'],data['job']['stage']),(200,'running','beam_rec'));self.assertNotIn('project',data)
            status,data=request(self.url+'/api/reconstruct/video/'+job['id'])
            self.assertEqual((status,data['job']['status']),(200,'done'))
            p=data['project'];self.assertEqual(p['name'],'План по видео · Комната 1');self.assertEqual(len(p['walls']),4)
            self.assertEqual(p['provenance']['service']['job'],job['id']);self.assertEqual(p['provenance']['alignment']['floorSource'],'cloud-density-peaks')
            self.assertTrue(p['pointCloud']);validate_project(p)
            # Scene bundle went to the shared space, referenced by the project; a repeated poll does not fetch it again.
            self.assertEqual((p['scenes'][0]['id'],p['scenes'][0]['frames'],p['scenes'][0]['source']['job']),(job['id'],2,job['id']))
            self.assertTrue(scenes.bundle_path(job['id']).is_file());self.assertEqual(FakeService.scene_downloads-downloads0,1)
            request(self.url+'/api/reconstruct/video/'+job['id']);self.assertEqual(FakeService.scene_downloads-downloads0,1)

    def test_job_journal_lists_jobs_and_reopens_results_later(self):
        with self.env():
            _,data=request(self.url+'/api/reconstruct/video',self.video('Кухня.mp4'));jid=data['job']['id']
            meta=jobs.get(jid);self.assertEqual((meta['kind'],meta['input']['name']),('video-reconstruction','Кухня.mp4'))
            self.assertIn(b'ftypmp42video-bytes',(jobs.job_dir(jid)/meta['input']['file']).read_bytes())
            other='20261020-000000-elsewh';FakeService.jobs[other]={'id':other,'status':'queued','stage':None,'message':None,'error':None,'files':[],
                                                                  'video':{'name':'android.mp4','size':1},'created':1,'polls':0}
            status,listing=request(self.url+'/api/jobs');self.assertEqual((status,listing['serviceError']),(200,None))
            by={j['id']:j for j in listing['jobs']}
            self.assertEqual((by[jid]['origin'],by[jid]['status'],by[jid]['hasInput']),('app','queued',True))
            self.assertEqual((by[other]['origin'],by[other]['video']['name']),('service','android.mp4'))  # started elsewhere, still visible
            for _ in range(2): _,data=request(self.url+'/api/reconstruct/video/'+jid)  # "another tab": no browser state at all
            self.assertEqual([s['name'] for s in data['project']['sources']],['Кухня.mp4'])   # recorded video is the source
            self.assertTrue(base64.b64decode(data['project']['sources'][0]['data']).endswith(b'video-bytes'))
            by={j['id']:j for j in request(self.url+'/api/jobs')[1]['jobs']}
            self.assertEqual((by[jid]['resultReady'],by[jid]['hasScene'],by[jid]['sceneStored']),(True,True,True))
            del FakeService.jobs[other]
        with self.env(ROOMPLAN_VIDEO_API_URL='http://127.0.0.1:9'):  # service down: the journal is still listed
            listing=request(self.url+'/api/jobs')[1]
            self.assertIn('недоступен',listing['serviceError']);self.assertIn(jid,[j['id'] for j in listing['jobs'] if j['status']=='unknown'])
        with self.assertRaises(ValueError): jobs.job_dir('../x')

    def test_failed_job_is_reported_without_project(self):
        with self.env():
            _,data=request(self.url+'/api/reconstruct/video',self.video('fail.mp4'))
            status,data=request(self.url+'/api/reconstruct/video/'+data['job']['id'])
            self.assertEqual((status,data['job']['status'],data['job']['error']),(200,'failed','poses: too few frames'));self.assertNotIn('project',data)

    def test_configuration_token_and_input_errors(self):
        with patch.dict(os.environ,{'ROOMPLAN_VIDEO_API_URL':'','ROOMPLAN_VIDEO_API_TOKEN':''}):
            self.assertFalse(health()['providers']['video']['available'])
            status,data=request(self.url+'/api/reconstruct/video',self.video());self.assertEqual(status,503);self.assertIn('ROOMPLAN_VIDEO_API_URL',data['error'])
            self.assertIn('disabled',health()['providers']['video']['status'])
        with self.env(ROOMPLAN_VIDEO_API_TOKEN='wrong'):
            status,data=request(self.url+'/api/reconstruct/video',self.video());self.assertEqual(status,503);self.assertIn('токен',data['error'])
        with self.env(ROOMPLAN_VIDEO_API_OPTIONS='[1]'): self.assertFalse(health()['providers']['video']['available'])
        with self.env(ROOMPLAN_VIDEO_API_URL='http://plan.example.org:8802'):  # token must not travel in clear
            video=health()['providers']['video'];self.assertFalse(video['available']);self.assertIn('https',video['status'])
            status,data=request(self.url+'/api/reconstruct/video',self.video());self.assertEqual(status,503);self.assertIn('https',data['error'])
        with self.env(ROOMPLAN_VIDEO_API_URL='https://plan.example.org:8443'): self.assertTrue(health()['providers']['video']['available'])
        with self.env():
            self.assertEqual(request(self.url+'/api/reconstruct/video',{'files':[]})[0],400)
            self.assertEqual(request(self.url+'/api/reconstruct/video',{'files':[{'name':'plan.png','mime':'image/png','data':'AAAA'}]})[0],400)
            self.assertEqual(request(self.url+'/api/reconstruct/video/..%2Fetc')[0],400)
            self.assertEqual(request(self.url+'/api/reconstruct/video/unknown-job')[0],400)
        with self.env(ROOMPLAN_VIDEO_API_URL='http://127.0.0.1:9'):
            self.assertEqual(request(self.url+'/api/reconstruct/video/some-job')[0],503)

class EnvFileTests(unittest.TestCase):
    def test_env_file_does_not_override_environment(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'RP_TEST_KEEP':'from-env'}):
            path=Path(d)/'.env';path.write_text('# comment\nRP_TEST_NEW="quoted value"\nRP_TEST_KEEP=from-file\nbroken line\n',encoding='utf-8')
            load_env(path);load_env(Path(d)/'missing.env')
            self.assertEqual(os.environ['RP_TEST_NEW'],'quoted value');self.assertEqual(os.environ['RP_TEST_KEEP'],'from-env')
            del os.environ['RP_TEST_NEW']

if __name__=='__main__': unittest.main()
