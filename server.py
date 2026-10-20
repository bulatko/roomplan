"""RoomPlan development server. Loopback by default; no installation/build step."""
import argparse
import json
import mimetypes
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
from roomplan.model import CATALOG, demo_project, validate_project
from roomplan.reconstruction import reconstruct
from roomplan.storage import load_project, save_project
from roomplan.export import export_svg
from roomplan.providers import health, reconstruct_object, ProviderUnavailable
from roomplan.video import encode_walkthrough
from roomplan.video_service import poll_video, submit_video

ROOT=Path(__file__).resolve().parent
MAX_REQUEST=360*1024*1024
VIDEO_JOB_PREFIX='/api/reconstruct/video/'

def load_env(path):
    """KEY=VALUE lines from a local .env; variables already set in the environment win."""
    if not path.is_file(): return
    for line in path.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        key,value=line.split('=',1);key=key.strip();value=value.strip()
        if len(value)>=2 and value[0]==value[-1] and value[0] in '"\'': value=value[1:-1]
        if key and key not in os.environ: os.environ[key]=value

class Handler(BaseHTTPRequestHandler):
    server_version='RoomPlan/1.0'
    def reply(self,status,data,mime='application/json; charset=utf-8',filename=None):
        if not isinstance(data,bytes): data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-cache')
        if filename: self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers();self.wfile.write(data)
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path=='/api/demo': return self.reply(200,demo_project())
        if path=='/api/catalog': return self.reply(200,CATALOG)
        if path=='/api/health': return self.reply(200,health())
        if path.startswith(VIDEO_JOB_PREFIX):
            try:
                result=poll_video(path.removeprefix(VIDEO_JOB_PREFIX))
                if 'project' in result:
                    result['project'],v=validate_project(result['project']);result['warnings']+=v
                return self.reply(200,result)
            except ProviderUnavailable as e: return self.reply(503,{'error':str(e)})
            except (ValueError,KeyError,TypeError,OverflowError) as e: return self.reply(400,{'error':str(e)})
            except Exception as e:
                self.log_error('%s',e);return self.reply(500,{'error':'Ошибка обработки. Подробности в локальном журнале сервера.'})
        if path.startswith('/api/'): return self.reply(404,{'error':'Неизвестный API маршрут'})
        if path=='/': candidate=ROOT/'web'/'index.html'
        elif path in ['/progress','/progress/']: candidate=ROOT/'web'/'progress.html'
        elif path=='/progress.json': candidate=ROOT/'progress.json'
        elif path.startswith(('/examples/','/docs/','/artifacts/reviews/','/artifacts/reference/','/artifacts/fixtures/')): candidate=ROOT/path.lstrip('/')
        else: candidate=ROOT/'web'/path.removeprefix('/web/').lstrip('/')
        candidate=candidate.resolve()
        allowed=[ROOT/'web',ROOT/'examples',ROOT/'docs',ROOT/'artifacts'/'reviews',ROOT/'artifacts'/'reference',ROOT/'artifacts'/'fixtures']
        if candidate!=ROOT/'progress.json' and not any(candidate.is_relative_to(p) for p in allowed): return self.reply(403,{'error':'Доступ запрещён'})
        if not candidate.is_file(): return self.reply(404,{'error':'Файл не найден'})
        if candidate.suffix.lower() in ['.tar','.gz','.tgz']: return self.reply(403,{'error':'Архив не опубликован'})
        mime=mimetypes.guess_type(candidate.name)[0] or 'application/octet-stream'
        if candidate.suffix=='.js': mime='text/javascript'
        self.reply(200,candidate.read_bytes(),mime)
    def do_POST(self):
        # No cross-origin mutation: native localhost app; reject foreign browser origins.
        origin=self.headers.get('Origin')
        if origin and urlsplit(origin).netloc!=self.headers.get('Host'): return self.reply(403,{'error':'Cross-origin requests are disabled'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=MAX_REQUEST: return self.reply(413,{'error':'Пустой запрос или превышен лимит 360 МБ'})
            payload=json.loads(self.rfile.read(length))
            if not isinstance(payload,dict): raise ValueError('Ожидается JSON-объект')
            path=urlsplit(self.path).path
            if path=='/api/reconstruct':
                p,w=reconstruct(payload);p,v=validate_project(p);return self.reply(200,{'project':p,'warnings':w+v})
            if path=='/api/reconstruct/object': return self.reply(200,reconstruct_object(payload))
            if path=='/api/reconstruct/video': return self.reply(200,submit_video(payload))
            if path=='/api/validate':
                p,w=validate_project(payload.get('project'));return self.reply(200,{'project':p,'warnings':w})
            if path=='/api/export/project': return self.reply(200,save_project(payload.get('project')),'application/zip','roomplan-project.zip')
            if path=='/api/import/project':
                p,w=load_project(payload.get('data',''));return self.reply(200,{'project':p,'warnings':w})
            if path=='/api/export/plan': return self.reply(200,export_svg(payload.get('project')),'image/svg+xml; charset=utf-8','roomplan-furnished-plan.svg')
            if path=='/api/export/video': return self.reply(200,encode_walkthrough(payload),'video/webm','roomplan-walkthrough.webm')
            return self.reply(404,{'error':'Неизвестный API маршрут'})
        except ProviderUnavailable as e: return self.reply(503,{'error':str(e)})
        except (ValueError,KeyError,TypeError,OverflowError) as e: return self.reply(400,{'error':str(e)})
        except Exception as e:
            self.log_error('%s',e);return self.reply(500,{'error':'Ошибка обработки. Подробности в локальном журнале сервера.'})

def main():
    load_env(ROOT/'.env')
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'RoomPlan: http://{args.host}:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
