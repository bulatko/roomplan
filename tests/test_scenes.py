import io
import json
import os
import tarfile
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from roomplan import scenes
from roomplan.model import demo_project
from roomplan.storage import load_project, save_project
from server import Handler

def make_bundle(top='scene_job1',fmt='roomplan-scene/1',extra=(),drop=()):
    """Minimal roomplan-scene/1 bundle: poses.json, two images, cloud.ply, plan.json."""
    poses={'format':fmt,'world':{'note':'metres, Z up, floor z=0'},'counts':{'cloud_points':1234},'checks':{'ok':True},
           'frames':[{'name':'000001.jpg','T_cw':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},{'name':'000002.jpg','T_cw':[[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]}]}
    files={'poses.json':json.dumps(poses).encode(),'images/000001.jpg':b'\xff\xd8jpg1','images/000002.jpg':b'\xff\xd8jpg2',
           'cloud.ply':b'ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n','plan.json':b'{"walls":[]}'}
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as tf:
        for name,data in files.items():
            if name in drop: continue
            info=tarfile.TarInfo(f'{top}/{name}');info.size=len(data);tf.addfile(info,io.BytesIO(data))
        for info,data in extra: tf.addfile(info,io.BytesIO(data) if data is not None else None)
    return out.getvalue()

class SceneStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.env=patch.dict(os.environ,{'ROOMPLAN_DATA_DIR':self.tmp.name});self.env.start()
    def tearDown(self):
        self.env.stop();self.tmp.cleanup()

    def test_store_and_summary(self):
        raw=make_bundle()
        info=scenes.store_bundle('job1',raw,source={'service':'roomplan-app','job':'job1'})
        self.assertEqual((info['frames'],info['images'],info['cloudPoints'],info['hasPlan'],info['checksOk'],info['bytes']),(2,2,1234,True,True,len(raw)))
        self.assertEqual(scenes.summary('job1'),info);self.assertEqual(scenes.bundle_path('job1').read_bytes(),raw)
        self.assertIsNone(scenes.summary('other'))

    def test_rejects_unsafe_or_foreign_bundles(self):
        evil=tarfile.TarInfo('scene_job1/../../evil');evil.size=1
        link=tarfile.TarInfo('scene_job1/images/link');link.type=tarfile.SYMTYPE;link.linkname='/etc/passwd'
        other=tarfile.TarInfo('elsewhere/x');other.size=1
        for raw in (make_bundle(extra=[(evil,b'x')]),make_bundle(extra=[(link,None)]),make_bundle(extra=[(other,b'x')]),
                    make_bundle(drop=('poses.json',)),make_bundle(fmt='other/1'),b'not a tar'):
            with self.assertRaises(ValueError): scenes.store_bundle('job1',raw)
        self.assertFalse(scenes.bundle_path('job1').exists())
        for bad in ('../x','a/b','',None,'x'*101):
            with self.assertRaises(ValueError): scenes.scene_dir(bad)

    def test_portable_zip_carries_the_bundle(self):
        raw=make_bundle();info=scenes.store_bundle('job1',raw)
        p=demo_project();p['scenes']=[info]
        archive=save_project(p)
        with tempfile.TemporaryDirectory() as other, patch.dict(os.environ,{'ROOMPLAN_DATA_DIR':other}):
            import base64
            q,_=load_project(base64.b64encode(archive).decode())
            self.assertEqual(scenes.bundle_path('job1').read_bytes(),raw)
            self.assertEqual(q['scenes'][0]['id'],'job1');self.assertNotIn('archivePath',q['scenes'][0])
        p['scenes']=[info,dict(info)]
        with self.assertRaises(ValueError): save_project(p)  # duplicate scene ids

class SceneAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.env=patch.dict(os.environ,{'ROOMPLAN_DATA_DIR':cls.tmp.name});cls.env.start()
        cls.raw=make_bundle();scenes.store_bundle('job1',cls.raw)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join();cls.env.stop();cls.tmp.cleanup()
    def get(self,path,data=None):
        req=urllib.request.Request(self.url+path,data=data,headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req) as r: return r.status,r.headers,r.read()
        except urllib.error.HTTPError as e: return e.code,e.headers,e.read()

    def test_summary_bundle_and_detection_stub(self):
        status,_,body=self.get('/api/scenes/job1');self.assertEqual((status,json.loads(body)['frames']),(200,2))
        status,headers,body=self.get('/api/scenes/job1/bundle')
        self.assertEqual((status,body),(200,self.raw));self.assertIn('scene_job1.tar.gz',headers['Content-Disposition'])
        status,_,body=self.get('/api/scenes/job1/detect',b'{}')
        self.assertEqual(status,503);self.assertIn('не подключена',json.loads(body)['error'])  # explicit stub, nothing invented
        self.assertEqual(self.get('/api/scenes/nope')[0],404);self.assertEqual(self.get('/api/scenes/nope/detect',b'{}')[0],400)
        self.assertEqual(self.get('/api/scenes/..%2Fx')[0],400)
        status,_,body=self.get('/api/health');self.assertEqual(json.loads(body)['providers']['detection']['mode'],'stub')

if __name__=='__main__': unittest.main()
