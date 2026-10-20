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
from roomplan.model import validate_project
from roomplan.providers import health
from roomplan.video_service import dominant_angle, floor_ceiling, public_job, read_ply, result_to_project
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
        self.assertTrue(p['layers']['pointCloud']);self.assertLessEqual(len(p['pointCloud']),60000)
        cloud=np.array(p['pointCloud']);self.assertAlmostEqual(float(np.median(cloud[cloud[:,1]<1,1])),0,delta=.02)
        self.assertTrue(any('пропущены: 1' in x for x in w))

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
    token='secret';jobs={};uploads=[]
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
        m=re.fullmatch(r'/api/jobs/([^/]+)(?:/files/([^/]+))?',self.path)
        job=FakeService.jobs.get(m.group(1)) if m else None
        if not job: return self.send(404,{'detail':'no such job'})
        if m.group(2):
            if m.group(2)=='result.json': return self.send(200,synthetic_result())
            if m.group(2)=='cloud.ply': return self.send(200,ply_binary(synthetic_cloud()[::4]),'application/octet-stream')
            return self.send(404,{'detail':'no such file'})
        job['polls']+=1
        if job['id'].endswith('fail00'): job.update(status='failed',stage='beam_rec',error='poses: too few frames')
        elif job['polls']==1: job.update(status='running',stage='beam_rec',message='VGGT')
        else: job.update(status='done',stage='done',files=['bev.png','cloud.ply','result.json'])
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
    @classmethod
    def tearDownClass(cls):
        for server,thread in cls.servers: server.shutdown();server.server_close();thread.join()
    def env(self,**kw):
        values={'ROOMPLAN_VIDEO_API_URL':self.service,'ROOMPLAN_VIDEO_API_TOKEN':'secret','ROOMPLAN_VIDEO_API_OPTIONS':'{"beam_rec":{"target_frames":200}}'};values.update(kw)
        return patch.dict(os.environ,{k:v for k,v in values.items() if v is not None},clear=False)
    def video(self,name='room.mp4'): return {'files':[{'name':name,'mime':'video/mp4','data':base64.b64encode(b'\x00\x00\x00\x18ftypmp42video-bytes').decode()}]}

    def test_upload_poll_and_convert(self):
        with self.env():
            self.assertTrue(health()['providers']['video']['available'])
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
